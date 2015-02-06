import util
from pylab import *
import gpu_lock
import os
import cudamat_ext as CM
import GaussianBinaryRBM, BinaryBinaryRBM
import sys, pdb
import dbn_data_src
from subprocess import call
from google.protobuf.text_format import MessageToString, Merge
import dbn_pb2
import logging
logger = logging.getLogger('dbn')
logger.setLevel(logging.INFO)
# create console handler and set level to info
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)


GAUSSIAN_BINARY_RBM=0
BINARY_BINARY_RBM=1

class dbn(object):
    def __init__(self, config_file):
        f = open(config_file)
        cfg_str = f.read()
        f.close()
        dbn_def = dbn_pb2.dbn()
        Merge(cfg_str, dbn_def)

        self.lst_configs = [config for config in dbn_def.layer_configs]
        self.num_layers = len(self.lst_configs)


    def project_data(self, data_src, dev_src, batch_size, rbm, 
                      file_name, config_def):
        logging.info("Will write projections to file: " + file_name)
        sys.stderr.write("Will write projections to file: %s\n"%file_name)

        cm_hidprobs = CM.empty((rbm.num_units, batch_size))
        next_data_src = dbn_data_src.data_src(file_name, rbm.num_units)

        for batch_data in data_src.get_iterator(batch_size, return_labels=False):
            rbm.project(batch_data, cm_hidprobs)
            next_data_src.add_data(cm_hidprobs.asarray())
        next_data_src.finish_adding()

        dev_file = file_name + ".dev"
        logging.info("Will write dev projections to file: " + dev_file)
        sys.stderr.write("Will write dev projections to file: %s\n"%dev_file)
        next_dev_src = dbn_data_src.data_src(dev_file, rbm.num_units)
        for batch_data in dev_src.get_iterator(batch_size, return_labels=False):
            rbm.project(batch_data, cm_hidprobs)
            next_dev_src.add_data(cm_hidprobs.asarray())
        next_dev_src.finish_adding()


        return next_data_src, next_dev_src
 
    def train(self, data_src, dev_src, batch_size, 
                 results_folder, reload_if_exists=False):
        logging.info("Number of layers in dbn = %d"%len(self.lst_configs))
        logging.info("Will save DBN parameters in folder: %s"%results_folder)
        sys.stderr.write("Number of layers in dbn = %d\n"%len(self.lst_configs))
        sys.stderr.write("Will save DBN parameters in folder: %s\n"%results_folder)

        layer_fldr = os.path.join(results_folder, "layers")
        projection_fldr = os.path.join(results_folder, "projections")
        if not os.path.exists(layer_fldr): os.makedirs(layer_fldr)

        if self.lst_configs[0].input_dim == -1:
            self.lst_configs[0].input_dim = data_src.get_data_dim()

        for (layer_num, config_def) in enumerate(self.lst_configs):
            param_file = os.path.join(layer_fldr, str(layer_num) + ".dat")

            logging.info("Training layer #%d of dim (%d,%d) for %d epochs"%(\
                          layer_num,config_def.input_dim, config_def.num_units,\
                          config_def.num_epochs))
            sys.stderr.write("Training layer #%d of dim (%d,%d) for %d epochs\n"%(\
                          layer_num,config_def.input_dim, config_def.num_units,\
                          config_def.num_epochs))
                                  
            if config_def.type == dbn_pb2.LayerConfig.GAUSSIAN_BINARY:
                logging.info( "Type: Gaussian-Binary RBM")
                sys.stderr.write( "Type: Gaussian-Binary RBM.\n")
                if reload_if_exists and os.path.exists(param_file):
                    sys.stderr.write('Reloading existing wts\n')
                    logging.info('Reloading existing wts')
                    rbm = GaussianBinaryRBM.GaussianBinaryRBM(\
                                           param_file = param_file)
                else:
                    rbm = GaussianBinaryRBM.GaussianBinaryRBM(\
                                                  config_def = config_def)
                    rbm.train(data_src, dev_src, config_def, batch_size, 
                              param_file)
            elif config_def.type == dbn_pb2.LayerConfig.BINARY_BINARY:
                logging.info( "Type: Binary-Binary RBM.")
                sys.stderr.write( "Type: Binary-Binary RBM.\n")
                if reload_if_exists and os.path.exists(param_file):
                    logging.info('Reloading existing wts')
                    sys.stderr.write('Reloading existing wts\n')
                    rbm = BinaryBinaryRBM.BinaryBinaryRBM(\
                                                   param_file = param_file)
                else:
                    rbm = BinaryBinaryRBM.BinaryBinaryRBM(config_def = config_def)
                    rbm.train(data_src, dev_src, config_def, batch_size, 
                              param_file)
            else:
                raise ArgumentException, "Unexpected layer type"


            if layer_num != self.num_layers-1:
                if not os.path.exists(projection_fldr):
                    os.makedirs(projection_fldr)
                data_file = os.path.join(projection_fldr, 
                                         str(layer_num+1) + "_data.dat")
                sys.stderr.write("Projecting data to next layer.\n")
                logging.info("Projecting data to next layer.")
                data_src, dev_src = self.project_data(data_src, dev_src, 
                                batch_size, rbm, data_file, config_def)

            if layer_num > 0:
                # delete the data that was created.
                sys.stderr.write("Removing projection file:%s\n"%data_file_last)
                logging.info("Removing projection file: %s"%data_file_last)
                #os.remove(data_file_last)
                call(['rm', data_file_last])
                dev_file_last = data_file_last + ".dev"
                logging.info("Removing dev projection file: %s"%dev_file_last)
                sys.stderr.write("Removing dev projection file: %s\n"%dev_file_last)
                call(['rm', dev_file_last])

            data_file_last = data_file
