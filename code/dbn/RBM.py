import util
from pylab import ones, zeros, sqrt, log, Inf
import cudamat_ext as cm
import time, pdb
import logging
logger = logging.getLogger('rbm')
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


class RBM(object):
    def __init__(self, param_file=None, config_def = None):
        if param_file is not None and config_def is not None:
            raise ArgumentException, "both param_file and \
                                  param cannot be passed in"
        if config_def is not None:
            self.init_from_config(config_def)
        if param_file is not  None:
            self.load_params(param_file)

    def project(self, cm_data, cm_hidprobs):
        cm.dot(self.cmW.T, cm_data, cm_hidprobs)
        cm_hidprobs.add_col_vec(self.cmBiasesHid)            
        cm_hidprobs.apply_sigmoid()

    def init_from_config(self, config_def):
        self.input_dim = config_def.input_dim
        self.num_units = config_def.num_units

        self.cmW = None
        self.cmW = cm.empty((config_def.input_dim, config_def.num_units))
        self.cmW.fill_with_randn()
        self.cmW.mult(config_def.wt_sigma)
       
        self.cmBiasesVis = None 
        self.cmBiasesVis = cm.empty((config_def.input_dim, 1))
        self.cmBiasesVis.assign(config_def.vis_bias)

        self.cmBiasesHid = None
        self.cmBiasesHid = cm.empty((config_def.num_units, 1))
        self.cmBiasesHid.assign(config_def.hid_bias)

    def create_gradients(self):
        self.cmWInc = cm.empty(self.cmW.shape)
        self.cmWInc.assign(0)
         
        self.cmBiasesHidInc = cm.empty(self.cmBiasesHid.shape)
        self.cmBiasesHidInc.assign(0)

        self.cmBiasesVisInc = cm.empty(self.cmBiasesVis.shape)
        self.cmBiasesVisInc.assign(0)


    def free_gradients(self):
        self.cmWInc.free_device_memory()
        self.cmWInc = None
         
        self.cmBiasesHidInc.free_device_memory()
        self.cmBiasesHidInc = None

        self.cmBiasesVisInc.free_device_memory()
        self.cmBiasesVisInc = None


    def get_num_units(self):
        return self.num_units

    def load_params(self, param_file):
        targetDict = {}
        util.load(param_file, targetDict, verbose=False)
        self.cmW = cm.CUDAMatrix(cm.reformat(targetDict['W']))
        self.cmBiasesHid = cm.CUDAMatrix(cm.reformat(targetDict['biasesHid']))
        self.cmBiasesVis = cm.CUDAMatrix(cm.reformat(targetDict['biasesVis']))

        self.input_dim, self.num_units = self.cmW.shape

    def save_params(self, param_file):
        util.save(param_file, 'W biasesHid biasesVis',
               {'W': self.cmW.asarray(),
                'biasesHid': self.cmBiasesHid.asarray(),
                'biasesVis': self.cmBiasesVis.asarray()})

    def allocate_activations(self, batch_size):
        cm_recon = cm.empty((self.input_dim, batch_size))
        cm_hidprobs = cm.empty((self.num_units, batch_size))
        cm_hidstates = cm.empty((self.num_units, batch_size))

        cm_posprods = cm.empty((self.input_dim, self.num_units))
        cm_negprods = cm.empty((self.input_dim, self.num_units))

        cm_poshidacts = cm.empty((self.num_units, 1))
        cm_neghidacts = cm.empty((self.num_units, 1))

        cm_posvisacts = cm.empty((self.input_dim, 1))
        cm_negvisacts = cm.empty((self.input_dim, 1))

        return cm_recon, cm_hidprobs, cm_hidstates, cm_posprods, cm_negprods, \
            cm_poshidacts, cm_neghidacts, cm_posvisacts, cm_negvisacts


    def train(self, data_src, dev_src, config_def, batch_size, param_file):
        self.create_gradients()

        eps_ratio = 1.0
        dev_err_min = Inf
        for self.epoch in range(config_def.num_epochs):
            if self.epoch >= config_def.mom_switch_epoch:
                momentum= config_def.final_momentum
            else:
                momentum= config_def.initial_momentum
            dev_err = self.train_cd1_for_epoch(data_src, dev_src, batch_size, 
                                      momentum, config_def.epsilon_w*eps_ratio, 
                                      config_def.epsilon_b*eps_ratio, 
                                      config_def.epsilon_b*eps_ratio, 
                                      config_def.l2_decay)
            if dev_err < dev_err_min:
                dev_err_min = dev_err
                self.save_params(param_file)
            elif eps_ratio > 0.5**4: 
                eps_ratio *= 0.5
                logging.info("Annealing learning rate by a factor of 2")
                self.load_params(param_file)
            else:
                eps_ratio *= 0.95

        self.free_gradients()

    def train_cd1_for_epoch(self, data_src, dev_src, momentum, epsilonw, 
                            epsilonvb, epsilonhb):
        raise Exception, "Should be called only for derived classes"

