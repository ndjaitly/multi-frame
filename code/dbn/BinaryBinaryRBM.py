import util
from numpy import ones, zeros, sqrt, log
import cudamat_ext as cm
import time, RBM
import logging, sys
logger = logging.getLogger('GaussianBinaryRBM')
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


class BinaryBinaryRBM(RBM.RBM):
    def __init__(self, param_file=None, config_def = None):
        super(BinaryBinaryRBM, self).__init__(param_file, config_def)

    def compute_recon_error_for_db(self, data_src, batch_size):
        err_sum, num_pts = 0, 0

        for batch_data in data_src.get_iterator(batch_size, return_labels=False):
            ######### START POSITIVE PHASE ############
            try: 
                cm_hidprobs
            except NameError:
                cm_hidprobs = cm.empty((self.num_units, batch_size))
                cm_recon = cm.empty(batch_data.shape).assign(0)

            cm.dot(self.cmW.T, batch_data, cm_hidprobs)
            cm_hidprobs.add_col_vec(self.cmBiasesHid)
            cm_hidprobs.apply_sigmoid()
         
            cm.dot(self.cmW, cm_hidprobs, target=cm_recon)
            cm_recon.add_col_vec(self.cmBiasesVis)
            cm_recon.apply_sigmoid()
            cm_recon.subtract(batch_data)
            err = cm_recon.euclid_norm()**2
            err_sum = err + err_sum
            num_pts = num_pts + batch_size
        return sqrt(err_sum*1./(self.input_dim*num_pts))


    def train_cd1_for_epoch(self, data_src, dev_src, batch_size, momentum, 
                            epsilonw, epsilonvb, epsilonhb, l2_decay):
        batch, err_sum, total_active, num_batch_cnts = 0, 0, 0, 0
        cm_recon, cm_hidprobs, cm_hidstates, cm_posprods, cm_negprods, \
          cm_poshidacts, cm_neghidacts, cm_posvisacts, cm_negvisacts = \
                     self.allocate_activations(batch_size)
        
        for batch_data in data_src.get_iterator(batch_size, return_labels=False):
            batch = batch + 1
            ######### START POSITIVE PHASE ############
            cm.dot(self.cmW.T, batch_data, cm_hidprobs)
            cm_hidprobs.add_col_vec(self.cmBiasesHid)            
            cm_hidprobs.apply_sigmoid()
            cm_hidstates.fill_with_rand()
            cm_hidstates.less_than(cm_hidprobs)
            num_active = cm_hidstates.euclid_norm()**2
        
            if batch % 20 == 0: 
                cm.dot(self.cmW, cm_hidprobs, target=cm_recon)
                cm_recon.add_col_vec(self.cmBiasesVis)
                cm_recon.apply_sigmoid()
                cm_recon.subtract(batch_data)
                err = cm_recon.euclid_norm()**2
                err_sum = err + err_sum
                num_batch_cnts = num_batch_cnts + 1

            cm.dot(batch_data, cm_hidprobs.T,  target=cm_posprods)
            cm_hidprobs.sum(axis=1, target=cm_poshidacts)
            batch_data.sum(axis=1, target=cm_posvisacts)
        
            ######### START NEGATIVE PHASE#########
            cm.dot(self.cmW, cm_hidstates, target=cm_recon)
            cm_recon.add_col_vec(self.cmBiasesVis)
            cm_recon.apply_sigmoid()
        
            cm.dot(self.cmW.T, cm_recon, target=cm_hidprobs)
            cm_hidprobs.add_col_vec(self.cmBiasesHid)
            cm_hidprobs.apply_sigmoid()
        
            cm.dot(cm_recon, cm_hidprobs.T, target=cm_negprods)
            cm_hidprobs.sum(axis=1, target=cm_neghidacts)
            cm_recon.sum(axis=1, target=cm_negvisacts)

            self.cmWInc.mult(momentum)
            cm_posprods.subtract(cm_negprods)
            self.cmWInc.add_mult(cm_posprods, epsilonw/batch_size)
        
            self.cmBiasesHidInc.mult(momentum)
            cm_poshidacts.subtract(cm_neghidacts)
            self.cmBiasesHidInc.add_mult(cm_poshidacts, epsilonhb/batch_size)
        
            self.cmBiasesVisInc.mult(momentum)
            cm_posvisacts.subtract(cm_negvisacts)
            self.cmBiasesVisInc.add_mult(cm_posvisacts, epsilonvb/batch_size)

            self.cmW.add_mult(self.cmW, -l2_decay*epsilonw)

            self.cmW.add(self.cmWInc)
            self.cmBiasesHid.add(self.cmBiasesHidInc)
            self.cmBiasesVis.add(self.cmBiasesVisInc)

            total_active = total_active + num_active


        avg_active = total_active/(batch*batch_size*self.num_units)
        avg_err = sqrt(err_sum/(num_batch_cnts*batch_size*self.input_dim))
        logging.info("Epoch # %d, batch %d, avg(err) %.4f, avg(on) = %.3f"%(\
                      self.epoch, batch, avg_err, avg_active))
        sys.stderr.write("Epoch # %d, batch %d, avg(err) %.4f, avg(on) = %.3f\n"%(\
                      self.epoch, batch, avg_err, avg_active))
        dev_error = self.compute_recon_error_for_db(dev_src, batch_size)
        logging.info("Epoch # %d, DEV SET avg(err) = %.4f"%(self.epoch,dev_error))
        sys.stderr.write("Epoch # %d, DEV SET avg(err) = %.4f\n"%(self.epoch,dev_error))
        return dev_error

