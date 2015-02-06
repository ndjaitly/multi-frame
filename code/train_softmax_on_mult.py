from pylab import *
import numpy, time
numpy.random.seed(int(time.time())) # This seed is meaningful. :).
import GPULock
id = GPULock.GetGPULock() 
import cudamat_ext as cm
cm.cublas_init()
cm.CUDAMatrix.init_random(int(time.time()))
import cudamat as cm2
cm2.CUDAMatrix.init_random(int(time.time()))
import pdb

def create_softmax_matrix(output_wts, biases, context_input, 
                          context_output, label_dim):
    num_frames_pt = output_wts.shape[1]/label_dim
    assert(label_dim * num_frames_pt == output_wts.shape[1])
    context_pt = num_frames_pt/2
    assert(num_frames_pt == 2 * context_pt + 1)
    assert(context_input <= context_pt)
    assert(context_input <= context_pt)

    num_units = output_wts.shape[0]
    num_input_frames = 2 * context_input + 1
    num_output_frames =  2 * context_output + 1
    # get the softx max wts and partition them.
    output_wts_new = zeros((num_units*num_input_frames, 
                            label_dim*num_output_frames))

    #start_frame = context_pt - context_input
    #end_frame = start_frame + 2 * context_inut + 1
    for frame in range(num_output_frames):
        s_out = frame * label_dim
        for in_frame in range(num_input_frames):
            frame_diff = (frame-context_output) - (in_frame-context_input) + context_pt
            if frame_diff < 0 or frame_diff >= num_frames_pt : continue
            wts = output_wts[:,(frame_diff*label_dim):((frame_diff+1)*label_dim)]
            s_in = in_frame * num_units
            output_wts_new[s_in:(s_in+num_units),s_out:(s_out+label_dim)] = wts


    return output_wts_new, tile(biases.reshape(label_dim, -1, order='F').sum(axis=1)[:,newaxis], (1, num_output_frames)).reshape(-1,1, order='F')

import logging
from mercurial import ui, hg, localrepo
import nnet_train_multi_soft, sys, os
import htkdb_mult_cm, shutil, util
import argparse
from helpers import compute_acc_mult
import run_params_pb2
from google.protobuf.text_format import Merge

parser = argparse.ArgumentParser()
parser.add_argument('--skip_repo', dest='skip_repo', 
                    action='store_true', default=False, 
                    help='Do not check repository state')
parser.add_argument('--skip_notes', dest='skip_notes', 
                    action='store_true', default=False, 
                    help='Do not ask for input notes')
parser.add_argument('--num_files_per_load', type=int, default=-1, 
         help='Number of files loaded to gpu each time (-1 for all)')
parser.add_argument('--skip_borders', type=int, default=0, 
         help='Number of border frames to skip')
parser.add_argument('--class_multipliers', type=str, default=None, 
         help='File with multipliers for gradients for each class type')
parser.add_argument('--dbn', type=str, default=None,
   help="Location of folder with DBN layers to be used for pretraining")
parser.add_argument('nn_def_file', help='Path to nn definition file')
parser.add_argument('param_def_file', help='Path to run params file')
parser.add_argument('db_name', help='Name of training database')
parser.add_argument('validation_db_name', 
                       help='Name of validation database')
parser.add_argument('db_path', help='Path to database')
parser.add_argument('output_fldr', type=str, help='output folder')

try:
    arguments = parser.parse_args()
except SystemExit, e:
    GPULock.FreeLock(id)
    sys.exit(e)

if not arguments.skip_repo:
    rep = localrepo.instance(ui.ui(), '.', False)
    if sum([len(x) for x in rep.status()]) != 0:
        print "Please commit changes to repository before running program"
        GPULock.FreeLock(id) 
        sys.exit(1)

if not os.path.exists(arguments.output_fldr):
    os.makedirs(arguments.output_fldr)
logPath = os.path.join(arguments.output_fldr, "log.txt")
if os.path.exists(logPath): os.remove(logPath)
rep = localrepo.instance(ui.ui(), '.', False)
revision_num = rep.changelog.headrevs()[0] 

# create logger
logging.basicConfig(filename=logPath, level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.info("python " + " ".join(sys.argv))

if not arguments.skip_notes:
    notes_line = raw_input("Please enter notes for this experiment " + \
                  " (blank line indices finished):")
    if not notes_line: 
        logging.info("NOTES: <None provided>")
    while (notes_line != ""):
        logging.info("NOTES: " + notes_line)
        notes_line = raw_input()
else:
    logging.info("NOTES: Logging skipped")


logging.info("Revision number for code: " + str(revision_num))
output_file = os.path.join(arguments.output_fldr, "model.dat")
shutil.copy2(arguments.nn_def_file,
             os.path.join(arguments.output_fldr, "nn_def.txt"))
shutil.copy2(arguments.param_def_file,
             os.path.join(arguments.output_fldr, "run_params.txt"))

class_multipliers = None
if arguments.class_multipliers is not None:
    class_multipliers = genfromtxt(arguments.class_multipliers)
    shutil.copy2(arguments.class_multipliers,
             os.path.join(arguments.output_fldr, "class_multipliers.txt"))

run_params = run_params_pb2.params()
f = open(arguments.param_def_file)
cfg_str = f.read()
f.close()
Merge(cfg_str, run_params)

pars_dict = {}
util.load(os.path.join(arguments.db_path, arguments.db_name, 'model.dat'),
          pars_dict)
layer_name = pars_dict['lst_layer_names'][-1]
output_wts = pars_dict[layer_name + "_wts"]
output_b = pars_dict[layer_name + "_b"]

print "# of output frame per pt = %d"%run_params.data_params.num_output_frames_per_pt
data_src = htkdb_mult_cm.htkdb_cm(arguments.db_name, arguments.db_path, 
                        run_params.data_params.num_frames_per_pt,
                        run_params.data_params.num_output_frames_per_pt,
                        run_params.data_params.use_delta, 
                        skip_borders=arguments.skip_borders)
validation_src = htkdb_mult_cm.htkdb_cm(arguments.validation_db_name,
                        arguments.db_path,
                        run_params.data_params.num_frames_per_pt,
                        run_params.data_params.num_output_frames_per_pt,
                        run_params.data_params.use_delta)

nn_train = nnet_train_multi_soft.nn()
nn_train.create_nnet_from_def(arguments.nn_def_file, 
                              data_dim = data_src.get_data_dim(),
                              target_dim = data_src.get_label_dim(),
                              num_softmaxes = run_params.data_params.num_output_frames_per_pt)

wts, b = create_softmax_matrix(output_wts, output_b,
                            (run_params.data_params.num_frames_per_pt-1)/2, 
                            (run_params.data_params.num_output_frames_per_pt-1)/2, 
                            data_src.get_label_dim())
nn_train._lst_layers[-1]._wts.load_matrix(wts/run_params.data_params.num_frames_per_pt)
nn_train._lst_layers[-1]._b.load_matrix(b/run_params.data_params.num_frames_per_pt)

nn_train.create_activations_and_probs(run_params.batch_size)
min_err = Inf
num_attempts = 0

cmn = run_params.data_params.normalization == run_params_pb2.DataParams.CMN
cmvn = run_params.data_params.normalization == run_params_pb2.DataParams.CMVN
normalize = run_params.data_params.normalization == run_params_pb2.DataParams.GLOB_NORMALIZE
data_src.setup_data(arguments.num_files_per_load,
                    cmn, cmvn, normalize)
validation_src.setup_data(arguments.num_files_per_load,
                    cmn, cmvn, normalize)
err_tol = 0
eps_ratio=1.0
for i in range(run_params.max_epochs):
    nn_train.train_one_epoch(data_src, eps_ratio, 
                              class_multipliers=class_multipliers)
    acc = compute_acc_mult(nn_train, validation_src)
    err = 100-acc
    logging.info("Dev set FER = %.4f"%err)
    sys.stderr.write("Dev set FER = %.4f\n"%err)
    sys.stderr.flush()
    if err <= min_err:
        nn_train.save(output_file)
        min_err = err
    elif err > min_err + err_tol and eps_ratio > (0.5**6):
        logging.info("Halving learning rate to %.4f. # of attempts = %d"%(\
                                                   eps_ratio, num_attempts))
        nn_train.load(output_file)
        nn_train.reset_momentums()
        eps_ratio *= 0.5
        num_attempts += 1
        err_tol = 0
    elif err > min_err + err_tol:
        eps_ratio *= 0.95

    data_src.permute_file_indices_for_loading()

GPULock.FreeLock(id) 
