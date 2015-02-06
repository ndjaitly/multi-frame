import cudamat_ext as cm
from numpy import sqrt, isnan, Inf
import pdb, nn_pb2, logging, sys

class layer(object):
    def __init__(self, name):
        self.name = name
        self.__num_grads = 0
        self.dropout = 0
        self._wt_mask = None

    @property
    def shape(self):
        return self._wts.shape

    @property
    def num_hid(self):
        return self._wts.shape[1]

    @property
    def num_dims(self):
        return self._wts.shape[0]

    def set_input_mask(self, mask):
        self._wt_mask = cm.CUDAMatrix(mask)


    def create_from_params(self, layer_def):
        sys.stderr.write("Initializing  layer: %s of type %s\n"%\
                      (layer_def.name, type(self).__name__))
        logging.info("Initializing  layer: %s of type %s\n"%\
                      (layer_def.name, type(self).__name__))
 
        init_params = layer_def.init_params
        self._wts = cm.empty((layer_def.input_dim, 
                              layer_def.num_units))
        self._wts.fill_with_randn()
        self._wts.mult(init_params.wt_sigma*1./sqrt(layer_def.input_dim))

        self._b = cm.empty((self._wts.shape[1], 1))
        self._b.fill_with_rand().mult(init_params.biases_max-\
                init_params.biases_min).add(init_params.biases_min)

        self._wts_grad = cm.empty(self._wts.shape).assign(0)
        self._wts_inc = cm.empty(self._wts.shape).assign(0)

        self._b_grad = cm.empty(self._b.shape).assign(0)
        self._b_inc = cm.empty(self._b.shape).assign(0)


    def add_params_to_dict(self, params_dict):
        params_dict[self.name + "_wts"] = self._wts.asarray()
        params_dict[self.name + "_b"] = self._b.asarray()

    def copy_params_from_dict(self, params_dict):
        self._wts = cm.CUDAMatrix(params_dict[self.name + "_wts"])
        self._b = cm.CUDAMatrix(params_dict[self.name + "_b"])

    def copy_from(self, src):
        self._wts.assign(src._wts)
        self._b.assign(src._b)

    def copy_transposed(self, src):
        src._wts.transpose(self._wts)

    def add_gradients(self, src):
        self._wts_grad.add(src._wts_grad)
        self._b_grad.add(src._b_grad)

    def add_gradients_transposed(self, src):
        try: self._wts_grad_cpy
        except AttributeError:
            self._wts_grad_cpy = cm.empty(self._wts.shape)
        src._wts_grad.transpose(self._wts_grad_cpy)
        self._wts_grad.add(self._wts_grad_cpy)

    def reset_momentums(self):
        self._wts_inc.assign(0)
        self._b_inc.assign(0)
        self.__num_grads = 0
 
    def apply_gradients(self, momentum, eps, l2=.0001, ada_lambda=0.95):
        if isnan(self._wts_grad.euclid_norm()):
            pdb.set_trace()
        self._wts_inc.mult(momentum)
        self._wts_inc.add_mult(self._wts_grad, eps)
        self._wts.add(self._wts_inc)
        self._wts.add_mult(self._wts, -l2*eps)
  
        if self._wt_mask is not None:
            self._wts.mult(self._wt_mask)
            self._wts_inc.mult(self._wt_mask)

        self._b_inc.mult(momentum)
        self._b_inc.add_mult(self._b_grad, eps)
        self._b.add(self._b_inc)


    def apply_wt_constraint(self, constraint):
        self._wts.threshold_column_norms(constraint, self._b)
        
    def project(self, cm_data, cm_act):
        cm.dot(self._wts.T, cm_data, cm_act)
        if self.dropout != 0:
            cm_act.mult(1./(1.-self.dropout))
        cm_act.add_col_vec(self._b)

    def back_prop(self, cm_act_grad, cm_data, cm_input_grad=None):
        ''' 
        back prop activation grad, and compute gradients. 
        '''
        cm.dot(cm_data, cm_act_grad.T, self._wts_grad)
        if cm_input_grad is not None:
            cm.dot(self._wts, cm_act_grad, cm_input_grad)
        cm_act_grad.sum(axis=1, target=self._b_grad)

        self._wts_grad.divide(cm_data.shape[1])
        self._b_grad.divide(cm_data.shape[1])
 

class linear_layer(layer):
    pass

    def fwd_prop(self, cm_data, cm_act, cm_output):
        if self.dropout != 0:
            cm_data.dropout(self.dropout)
        self.project(cm_data, cm_act)
        cm_output.assign(cm_act)

    def compute_act_grad_from_output_grad(self, cm_output, 
                                          cm_out_grad, cm_act_grad):
        cm_act_grad.assign(cm_out_grad)

    def compute_gradients_from_targets(self, cm_targets, 
                                       cm_output, cm_act, 
                                       cm_out_grad, cm_act_grad):
        cm_targets.subtract(cm_output, cm_out_grad)
        cm_act_grad.assign(cm_out_grad)


class sigmoid_layer(layer):
    pass 

    def fwd_prop(self, cm_data, cm_act, cm_output):
        if self.dropout != 0:
            cm_data.dropout(self.dropout)
        self.project(cm_data, cm_act)
        cm_act.apply_sigmoid(cm_output)
        #cm_output.greater_than(cm_probs)

    def compute_act_grad_from_output_grad(self, cm_output, 
                                    cm_out_grad, cm_act_grad):
        cm.compute_logistic_grad(cm_output, cm_out_grad, 
                                          target=cm_act_grad)

    def compute_gradients_from_targets(self, cm_targets, 
                                       cm_output, cm_act, 
                                       cm_out_grad, cm_act_grad):
        cm_targets.subtract(cm_output, cm_act_grad)

 
class softmax_layer(layer):
    pass

    def fwd_prop(self, cm_data, cm_act, cm_output):
        if self.dropout != 0:
            cm_data.dropout(self.dropout)
        self.project(cm_data, cm_act)
        cm_act.compute_softmax(cm_output)


    def compute_act_grad_from_output_grad(self, cm_output, 
                                    cm_out_grad, cm_act_grad):
        raise Exception, "softmax isn't really supposed to be used \
                            in internal layers"

    def compute_gradients_from_targets(self, cm_targets, 
                                       cm_output, cm_act, 
                                       cm_out_grad, cm_act_grad):
        cm_targets.subtract(cm_output, cm_act_grad)

 
class multi_softmax_layer(layer):
    pass

    def add_params_to_dict(self, params_dict):
        params_dict[self.name + "_wts"] = self._wts.asarray()
        params_dict[self.name + "_b"] = self._b.asarray()
        params_dict[self.name + "_num_softmaxes"] = self.num_softmaxes

    def copy_params_from_dict(self, params_dict):
        self._wts = cm.CUDAMatrix(params_dict[self.name + "_wts"])
        self._b = cm.CUDAMatrix(params_dict[self.name + "_b"])
        self.num_softmaxes = params_dict[self.name + "_num_softmaxes"]
        self.num_units = self.num_hid/self.num_softmaxes


    def create_from_params(self, layer_def):
        sys.stderr.write("Initializing  layer: %s of type %s\n"%\
                      (layer_def.name, type(self).__name__))
        logging.info("Initializing  layer: %s of type %s\n"%\
                      (layer_def.name, type(self).__name__))

        self.num_units = layer_def.num_units 
        self.num_softmaxes = layer_def.num_softmaxes 

        init_params = layer_def.init_params
        self._wts = cm.empty((layer_def.input_dim, 
              layer_def.num_units*layer_def.num_softmaxes))
        self._wts.fill_with_randn()
        self._wts.mult(init_params.wt_sigma*1./ sqrt(layer_def.input_dim))

        self._b = cm.empty((self._wts.shape[1], 1))
        self._b.fill_with_rand().mult(init_params.biases_max-\
                init_params.biases_min).add(init_params.biases_min)

        self._wts_grad = cm.empty(self._wts.shape).assign(0)
        self._wts_inc = cm.empty(self._wts.shape).assign(0)

        self._b_grad = cm.empty(self._b.shape).assign(0)
        self._b_inc = cm.empty(self._b.shape).assign(0)


    def fwd_prop(self, cm_data, cm_act, cm_output):
        if self.dropout != 0:
            cm_data.dropout(self.dropout)
        self.project(cm_data, cm_act)
        batch_size = cm_act.shape[1]
        cm_act.reshape((self.num_units, batch_size*self.num_softmaxes))
        cm_output.reshape((self.num_units, batch_size*self.num_softmaxes))
        cm_act.compute_softmax(cm_output)
        cm_act.reshape((self.num_units*self.num_softmaxes, batch_size))
        cm_output.reshape((self.num_units*self.num_softmaxes, batch_size))


    def compute_act_grad_from_output_grad(self, cm_output, 
                                    cm_out_grad, cm_act_grad):
        raise Exception, "softmax isn't really supposed to be used \
                            in internal layers"

    def compute_gradients_from_targets(self, cm_targets, 
                                       cm_output, cm_act, 
                                       cm_out_grad, cm_act_grad):
        cm_targets.subtract(cm_output, cm_act_grad)

 
class relu_layer(layer):
    pass

    def fwd_prop(self, cm_data, cm_act, cm_output):
        if self.dropout != 0:
            cm_data.dropout(self.dropout)
        self.project(cm_data, cm_act)
        cm_act.threshold_below(0,cm_output)

    def compute_act_grad_from_output_grad(self, cm_output, 
                                    cm_out_grad, cm_act_grad):
        cm_act_grad.assign(cm_output)
        cm_act_grad.greater_than(0)
        cm_act_grad.mult(cm_out_grad)

    def compute_gradients_from_targets(self, cm_targets, 
                                       cm_output, cm_act, 
                                       cm_out_grad, cm_act_grad):
        cm_targets.subtract(cm_output, cm_out_grad)
        cm_act_grad.assign(cm_output)
        cm_act_grad.greater_than(0)
        cm_act_grad.mult(cm_out_grad)
                              
def create_empty_nnet_layer(name, layer_type):
    if layer_type ==  nn_pb2.Layer.LINEAR:
        layer = linear_layer(name)
    elif layer_type == nn_pb2.Layer.SIGMOID:
        layer = sigmoid_layer(name)
    elif layer_type == nn_pb2.Layer.RELU:
        layer = relu_layer(name)
    elif layer_type == nn_pb2.Layer.SOFTMAX:
        layer = softmax_layer(name)
    elif layer_type == nn_pb2.Layer.MULTI_SOFTMAX:
        layer = multi_softmax_layer(name)
    else:
        raise Exception, "Unknown layer type"
    return layer

def create_nnet_layer(layer_def):
    layer = create_empty_nnet_layer(layer_def.name, layer_def.type)
    layer.create_from_params(layer_def)
    return layer
