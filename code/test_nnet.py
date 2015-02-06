from pylab import *
import numpy
numpy.random.seed(42) # This seed is meaningful. :).
from numpy.random import randn, multinomial
import GPULock
GPULock.GetGPULock() 
import cudamat_ext as cm
cm.cublas_init()
cm.CUDAMatrix.init_random(42)
import nnet_train, sys, os
import pdb
def init_data(lst_num_hid, data_dim, target_dim, 
                                    batch_size):
    data = randn(data_dim, batch_size)
    lst_wts, lst_hids = [], []
    input_dim = data_dim
    vis = data.copy()
    for layer_num, num_units in enumerate(lst_num_hid):
        # create wts
        wts = randn(input_dim, num_units)*.01
        # copy wts
        lst_wts.append(wts)
        # create hiddens
        if layer_num != len(lst_num_hid)-1:
            hid = 1./(1. + exp(-dot(wts.T, vis)))
        else:
            act = dot(wts.T, vis)
            act = act - act.max(axis=0).reshape(1,-1)
            hid = exp(act)/exp(act).sum(axis=0).reshape(1,-1)
        # copy hiddens
        lst_hids.append(hid)
        input_dim = num_units
        vis = hid.copy()

    # create targets from softmax probs
    targets = zeros((target_dim, batch_size))
    for j in range(batch_size):
        targets[:,j] = multinomial(1, hid[:,j], 1)

    # return data, targets, hiddens and wts
    return data, targets, lst_hids, lst_wts

def calc_lg_prob(data, targets, lst_wts):
    vis = data.copy()
    for layer_num, wts in enumerate(lst_wts):
        # create hiddens
        if layer_num != len(lst_wts)-1:
            hid = 1./(1. + exp(-dot(wts.T, vis)))
        else:
            act = dot(wts.T, vis)
            act = act - act.max(axis=0).reshape(1,-1)
            hid = exp(act)/exp(act).sum(axis=0).reshape(1,-1)
        vis = hid.copy()

    return sum(log(sum(vis*targets, axis=0)))

def set_nn_wts(nnet, lst_wts):
    for layer_num, (wts, layer) in enumerate(zip(lst_wts,
                               nnet._lst_layers)):
        layer._wts.load_matrix(wts)

def check_hid_activities(nnet, lst_hids):
    tol = 1e-6
    for layer_num, (hids_cpu, hids_gpu) in enumerate(zip(lst_hids,
                               nnet._lst_outputs)):
        err = ((hids_cpu - hids_gpu.asarray())**2).sum()
        if err > tol:
            sys.stdout.write("FAILED test for layer " + \
                             "%d, error = %.4f\n"%(layer_num,err))
        else:
            sys.stdout.write("Passed test for layer " + \
                             "%d, error = %.4f\n"%(layer_num,err))

def compare_grads(lst_grads_cpu, lst_grads_gpu):
    tol = 1e-6
    for layer_num, (grads_cpu, grads_gpu) in enumerate(zip(\
                                 lst_grads_cpu, lst_grads_gpu)):
        err = sqrt((((grads_cpu - grads_gpu)/(grads_cpu+1e-8))**2).mean())
        if err > tol:
            sys.stdout.write("FAILED test for layer " + \
                             "%d, error = %.4f\n"%(layer_num,err))
        else:
            sys.stdout.write("Passed test for layer " + \
                             "%d, error = %.4f\n"%(layer_num,err))


def cpu_grads(cm_data, cm_targets, nn_train, eps=1e-6):
    lst_wts = [layer._wts.asarray().copy() for layer in \
                                          nn_train._lst_layers]
    lst_wts_cpy = [x.copy() for x in lst_wts]

    lst_wts_grad = []

    data = cm_data.asarray().copy()
    targets = cm_targets.asarray().copy()
    lg_prob = calc_lg_prob(data, targets, lst_wts_cpy)

    for layer_num, (wts, wts_cpy) in enumerate(zip(lst_wts, 
                                               lst_wts_cpy)):
        wts_grad = zeros(wts.shape)
        for i in range(wts.shape[0]):
            for j in range(wts.shape[1]):
                wts[i,j] += eps
                lgprob_cur = calc_lg_prob(data, targets, lst_wts)
                wts[i,j] = wts_cpy[i,j]
                wts_grad[i,j] = (lgprob_cur - lg_prob)/eps/cm_data.shape[1]

        lst_wts_grad.append(wts_grad.copy())

    return lst_wts_grad

def gpu_grads(cm_data, cm_targets, nn_train, eps=1e-6):
    batch_size = cm_data.shape[1]
    cm_probs = cm.empty((1, batch_size))
    cm_correct = cm.empty((1, batch_size))

    nn_train.fwd_prop(cm_data)
    cm_predictions = nn_train._lst_outputs[-1]
    cm.compute_softmax_accuraccy(cm_predictions, 
                          cm_targets, cm_probs, cm_correct)
    lg_prob = sum(cm_probs.asarray())

    lst_grads = []
    for layer_num, (layer, wts) in enumerate(zip(
                       nn_train._lst_layers, lst_wts)):
        wts_grad = zeros(wts.shape)
        for i in range(wts.shape[0]):
            for j in range(wts.shape[1]):
                wts_cpy = wts.copy()
                wts_cpy[i,j] += eps
                layer._wts.load_matrix(wts_cpy)

                nn_train.fwd_prop(cm_data)
                cm.compute_softmax_accuraccy(cm_predictions, 
                          cm_targets, cm_probs, cm_correct)
                lg_prob_cur = sum(cm_probs.asarray())
                wts_grad[i,j] = (lg_prob_cur-lg_prob)/eps/batch_size

        # reset wt to original
        layer._wts.load_matrix(wts.copy())
        lst_grads.append(wts_grad.copy())
    return lst_grads


def check_lg_probs(cm_data, cm_targets, nn_train):
    lst_wts = [layer._wts.asarray().copy() for layer in \
                                          nn_train._lst_layers]
    lgprob_cpu = calc_lg_prob(cm_data.asarray().copy(), 
                                cm_targets.asarray().copy(), 
                                   lst_wts)

    batch_size = cm_data.shape[1]
    cm_probs = cm.empty((1, batch_size))
    cm_correct = cm.empty((1, batch_size))
    cm_predictions = nn_train._lst_outputs[-1]
    cm.compute_softmax_accuraccy(cm_predictions, cm_targets, 
                                   cm_probs, cm_correct)

    lgprob_gpu = sum(cm_probs.asarray())
    err = abs(lgprob_gpu - lgprob_cpu)/batch_size
    tol = 1e-6
    if err > tol:
        sys.stdout.write("FAILED test for log probs. " + \
             " cpu, gpu  = %.4f, %.4f\n"%(lgprob_cpu, lgprob_gpu))

    return lgprob_gpu, lgprob_cpu


data_dim = 5
target_dim = 15
batch_size=20
nn_def_file = "params/nn_def_20_10.txt"
nn_train = nnet_train.nn()
nn_train.create_nnet_from_def(nn_def_file,
                              data_dim = data_dim,
                              target_dim = target_dim)
nn_train.create_activations_and_probs(batch_size)

lst_num_hid = list(nn_train._lst_num_hid)
data, targets, lst_hids, lst_wts = init_data(lst_num_hid,
                                    data_dim, target_dim, 
                                    batch_size)
set_nn_wts(nn_train, lst_wts)
cm_data = cm.CUDAMatrix(data)
cm_targets = cm.CUDAMatrix(targets)


nn_train.fwd_prop(cm_data)
check_hid_activities(nn_train, lst_hids)

lgprob_orig, lgprob_orig_cpu = check_lg_probs(cm_data, cm_targets, 
                                                      nn_train)

nn_train.fwd_prop(cm_data)
cm_predictions = nn_train._lst_outputs[-1]
cm_targets.subtract(cm_predictions,
                       nn_train._lst_activations_grad[-1])
nn_train.back_prop(cm_data)
layer_grads = [layer._wts_grad.asarray().copy() for layer in \
                                         nn_train._lst_layers]

eps = 1e-4
lst_gpu_grads = gpu_grads(cm_data, cm_targets, nn_train, eps)
lst_cpu_grads = cpu_grads(cm_data, cm_targets, nn_train, eps)
compare_grads(layer_grads, lst_gpu_grads)
compare_grads(layer_grads, lst_cpu_grads)
compare_grads(lst_gpu_grads, lst_cpu_grads)
pdb.set_trace()
