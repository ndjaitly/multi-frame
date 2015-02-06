from nnet_layers import *
import nn_pb2, sys
from google.protobuf.text_format import Merge
import logging, util, os, pdb, copy
from numpy import zeros, array
 
# create logger
logger = logging.getLogger('nnet_train')
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


class nn(object):
    def __init__(self):
        pass

    def save(self, file_name):
        params_dict = {} 
        params_dict['lst_layer_names'] = [layer.name for layer in self._lst_layers]
        params_dict['lst_layer_type'] = self._lst_layer_type
        params_dict['lst_num_hid'] = self._lst_num_hid
        params_dict['data_dim'] = self._data_dim

        for layer in self._lst_layers:
            layer.add_params_to_dict(params_dict)

        util.save(file_name, " ".join(params_dict.keys()), params_dict)

    def reset_momentums(self):
        for layer in self._lst_layers:
            layer.reset_momentums()

    def load(self, file_name):
        params_dict = {}
        util.load(file_name, params_dict)
        if not hasattr(self, '_lst_layers'):
            self._lst_layer_type, self._lst_num_hid, self._data_dim = \
                                           params_dict['lst_layer_type'], \
                                               params_dict['lst_num_hid'], \
                                                   params_dict['data_dim']

            logging.info("Creating new layers from parameters in file: %s"%file_name)
            self._lst_layers = [] 
            for (layer_name, layer_type) in zip(params_dict['lst_layer_names'],
                                                self._lst_layer_type):
                layer = create_empty_nnet_layer(layer_name, layer_type)
                layer.copy_params_from_dict(params_dict)
                self._lst_layers.append(layer)
        else:
            logging.info("Updating layer parameters using file: %s"%file_name)
            for layer in self._lst_layers:
                layer.copy_params_from_dict(params_dict)

        self.num_layers = len(self._lst_layers)

    def copy_params_from_single_prediction_net(self, file_name):
        params_dict = {}
        util.load(file_name, params_dict)
        sys.stderr.write("Updating layer parameters from single time " + \
                         "nnet using file: %s\n"%file_name)
        logging.info("Updating layer parameters from single time nnet "+ \
                      "  using file: %s"%file_name)

        for layer_num in range(self.num_layers-1):
            sys.stderr.write("Updating layer # %d\n"%layer_num)
            sys.stderr.flush()
            logging.info("Updating layer # %d"%layer_num)
            self._lst_layers[layer_num].copy_params_from_dict(params_dict)


    def get_num_layers(self):
        return len(self._lst_layers)

    def get_code_dim(self):
        return self._lst_num_hid[-1]

    def load_from_dbn(self, dbn_fldr):
        logging.info("Using DBN for pretraining weights")
        for layer_num in range(len(self._lst_layers)-1):
            file_name = os.path.join(dbn_fldr, "layers", 
                                              str(layer_num) + ".dat")
            params_dict_in = {}
            if os.path.exists(file_name):
                logging.info("Using DBN to set parameters of layer %d"%layer_num)
                util.load(file_name, params_dict_in, verbose=False)
                layer_name = self._lst_layers[layer_num].name
                wts_name = layer_name + "_wts"
                biases_name = layer_name + "_b"
                params_dict_nnet = {wts_name: params_dict_in['W'], 
                                biases_name: params_dict_in['biasesHid']}
                self._lst_layers[layer_num].copy_params_from_dict(\
                                                      params_dict_nnet)
            else:
                raise Exception, "File does not exist for loading for DBN"

    def create_nnet_from_def(self, def_file, data_dim=-1, target_dim=-1, 
                             num_softmaxes=-1):
        f = open(def_file)
        cfg_str = f.read()
        f.close()

        nn_def = nn_pb2.nn()
        Merge(cfg_str, nn_def)

        self._layers = [] 
        self._nn_def = nn_def

        self.num_layers = len(nn_def.layers)

        self._data_dim = nn_def.layers[0].input_dim
        if data_dim != -1: self._data_dim = data_dim

        self._lst_num_hid = []
        self._lst_layer_type = []
        self._lst_layers = []

        assert(num_softmaxes != -1)
        nn_def.layers[-1].num_softmaxes = num_softmaxes
        for layer_num, layer_def in enumerate(nn_def.layers):
            if layer_num == 0:
                layer_def.input_dim = self._data_dim
            if layer_num == self.num_layers-1 and target_dim != -1:
                layer_def.num_units = target_dim

            self._lst_num_hid.append(layer_def.num_units)
            self._lst_layer_type.append(layer_def.type)
            layer = create_nnet_layer(layer_def)
                                      
            self._lst_layers.append(layer)

    def create_activations_and_probs(self, batch_size):
        self._lst_activations, self._lst_outputs = [], []
        self._lst_activations_grad, self._lst_outputs_grad = [], []

        self._batch_size = batch_size
        for layer_num, num_hid in enumerate(self._lst_num_hid):
            if layer_num == len(self._lst_num_hid)-1:
                num_hid *= self._lst_layers[-1].num_softmaxes

            self._lst_activations.append(cm.empty((num_hid, batch_size)))
            self._lst_outputs.append(cm.empty((num_hid, batch_size)))

            self._lst_activations_grad.append(cm.empty((num_hid, 
                                                        batch_size)))
            self._lst_outputs_grad.append(cm.empty((num_hid,
                                                    batch_size)))


    def predict(self, data, unnormalized=False):
        num_pts = 1000
        predictions = zeros((\
              self._lst_num_hid[-1]*self._lst_layers[-1].num_softmaxes, 
              data.shape[1]))

        index = 0
        for layer_num, layer in enumerate(self._lst_layers): 
            layer.dropout = 0
        while index < data.shape[1]:
            end_index = min(data.shape[1], index + num_pts)
            cur_preds = self.fwd_prop_np(data[:,index:end_index].copy(), 
                                                        unnormalized)
            predictions[:,index:end_index] = cur_preds
            index = end_index
        return predictions

    def fwd_prop_np(self, data, unnormalized, last_layer=-1):
        cm_input = cm.CUDAMatrix(data)
        num_pts = data.shape[1]

        if last_layer == -1: last_layer = self.num_layers-1
        for layer_num, layer in enumerate(self._lst_layers):
            act = cm.empty((layer.num_hid, num_pts))
            out = cm.empty((layer.num_hid, num_pts))

            layer.fwd_prop(cm_input, act, out)

            cm_input.free_device_memory()
            if layer_num == last_layer:   break
            if unnormalized == False:
                act.free_device_memory()
            cm_input = out

        if unnormalized:
            preds = act.asarray().copy() 
            act.free_device_memory()
            act = None
        else:
            preds = out.asarray().copy() 
        out.free_device_memory()
        out = None
        return preds
 
    def fwd_prop(self, cm_data):
        vis = cm_data
        for layer_num, layer  in enumerate(self._lst_layers):
            act, out = self._lst_activations[layer_num], \
                                    self._lst_outputs[layer_num]
            layer.fwd_prop(vis, act, out)
            vis = out

    def back_prop(self, cm_data):
        # now back propagate activation gradient.
        if self.num_layers != 1:
            self._lst_layers[-1].back_prop(self._lst_activations_grad[-1],
                                         self._lst_outputs[-2], \
                                         self._lst_outputs_grad[-2])
            self._lst_outputs_grad[-2].mult(1./(1.-self._lst_layers[-1].dropout))
        else:
            self._lst_layers[-1].back_prop(self._lst_activations_grad[-1],
                                           cm_data)
            return

        for layer_num in range(len(self._lst_layers)-2,0,-1):
            layer, cm_out, cm_act, cm_out_grad, cm_act_grad = \
                self._lst_layers[layer_num], self._lst_outputs[layer_num], \
                     self._lst_activations[layer_num], \
                        self._lst_outputs_grad[layer_num], \
                           self._lst_activations_grad[layer_num]
            layer.compute_act_grad_from_output_grad(cm_out, cm_out_grad,
                                                             cm_act_grad)
            layer.back_prop(cm_act_grad, self._lst_outputs[layer_num-1], \
                                         self._lst_outputs_grad[layer_num-1])
            self._lst_outputs_grad[layer_num-1].mult(\
                               1./(1.-layer.dropout))

        layer = self._lst_layers[0]
        layer.compute_act_grad_from_output_grad(self._lst_outputs[0], 
                                                self._lst_outputs_grad[0],
                                                self._lst_activations_grad[0])
        layer.back_prop(self._lst_activations_grad[0], cm_data)


    def apply_gradients(self, lst_schedules, eps_ratio):
        # need to add weight constraint if using dropouts.
        for (layer, schedule) in zip(self._lst_layers, lst_schedules):
            layer.apply_gradients(schedule.momentum, 
                                  eps_ratio*schedule.epsilon,
                                  l2=schedule.l2_wt, 
                                  ada_lambda=schedule.ada_lambda)
            if schedule.wt_norm_constraint != 0:
                layer.apply_wt_constraint(schedule.wt_norm_constraint*sqrt(layer.num_dims))

    def train_one_epoch(self, data_src, eps_ratio=1.0, class_multipliers=None):
        num_softmaxes = self._lst_layers[-1].num_softmaxes
        multi_softmax_shape = ((self._lst_layers[-1].num_units*num_softmaxes,
                                self._batch_size))
        softmax_shape = ((self._lst_layers[-1].num_units, 
                          num_softmaxes*self._batch_size))

        cm_probs = cm.empty((1, num_softmaxes*self._batch_size))
        cm_correct = cm.empty((1, num_softmaxes*self._batch_size))
        cm_predictions = self._lst_outputs[-1]
        cm_predictions_act_grad = self._lst_activations_grad[-1]

        cm_codes = self._lst_outputs[-1]
        cm_out_grad = self._lst_outputs_grad[-1]
        cm_ones = cm.empty(cm_probs.shape).assign(1)
        try:
            self.__cur_epoch += 1
        except AttributeError:
            self.__cur_epoch = 1

        try:
            self._tot_batch
        except AttributeError:
            self._tot_batch = 0


        # need to determine layer momentums and learning rates
        lst_layer_schedules = []
        reset_momentums = False
        for layer_num, layer_def in enumerate(self._nn_def.layers):
            for sched_num, learn_params_def in \
                           enumerate(layer_def.learning_schedule):
                if learn_params_def.start_epoch <= self.__cur_epoch and \
                     learn_params_def.end_epoch >= self.__cur_epoch:
                    lst_layer_schedules.append(learn_params_def)
                    if self.__cur_epoch == learn_params_def.start_epoch:
                        reset_momentums = True
                  
            self._lst_layers[layer_num].dropout = learn_params_def.dropout_rate
            if len(lst_layer_schedules) != layer_num + 1: 
                raise Exception, "Did not find relevant schedule " + \
                                 " for layer: " + str(layer_num)

        if reset_momentums: self.reset_momentums()
 
        num_pts, classif_err_sum, lg_p_sum = 0, 0, 0
        batch = 0

        
        printStr, printStrNew = '', ''

        for  (cm_data, cm_labels) in \
                 data_src.get_iterator(self._batch_size):
            batch += 1
            num_pts += self._batch_size * num_softmaxes
            self.fwd_prop(cm_data)
           
            cm_predictions.reshape(softmax_shape) 
            cm_labels.reshape(softmax_shape) 
            cm.compute_softmax_accuraccy(cm_predictions, cm_labels, 
                                         cm_probs, cm_correct)
            cm_predictions.reshape(multi_softmax_shape)
            cm_labels.reshape(multi_softmax_shape)

            classif_err_sum += (num_softmaxes*self._batch_size -cm.vdot(cm_correct, cm_ones))
            lg_p_sum += cm.vdot(cm_probs, cm_ones)
            if isnan(lg_p_sum):
                pdb.set_trace()

            cm_labels.subtract(cm_predictions, target=cm_predictions_act_grad)
            if class_multipliers is not None:
                labels = array(cm_labels.asarray().argmax(axis=0), 'int')
                class_mult = class_multipliers[labels].reshape((1,-1))
                cm_class_mult = cm.CUDAMatrix(class_mult)
                cm_predictions_act_grad.mult_by_row(cm_class_mult)

            self.back_prop(cm_data)
            self.apply_gradients(lst_layer_schedules, eps_ratio)
            self._tot_batch += 1
            if batch % 100 == 0:
                classif_err = classif_err_sum*100./num_pts
                printStr = "Epoch = %d, batch = %d, FER = %.3f, lg(p) %.4f, norm(wt) = %.4g"%(\
                                       self.__cur_epoch, batch, classif_err, lg_p_sum*1./num_pts,
                                       self._lst_layers[0]._wts.euclid_norm())
                printString = printStrNew + printStr
                sys.stderr.write(printString)
                sys.stderr.flush()
                printStrNew = '\b' * (len(printStr))


        classif_err = classif_err_sum*100./num_pts
        logging.info("Epoch = %d, batch = %d, FER = %.3f, lg(p) %.4f"%(\
                   self.__cur_epoch, batch, classif_err, lg_p_sum*1./num_pts))
        sys.stderr.write("Epoch = %d, batch = %d, FER = %.3f, lg(p) %.4f\n"%(\
                   self.__cur_epoch, batch, classif_err, lg_p_sum*1./num_pts))
        sys.stderr.flush()
        ch.flush()

