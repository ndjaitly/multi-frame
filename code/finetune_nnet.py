from pylab import *
import numpy, time
numpy.random.seed(int(time.time())) # This seed is meaningful. :).
import GPULock
id = GPULock.GetGPULock() 
import cudamat_ext as cm
cm.cublas_init()
cm.CUDAMatrix.init_random(int(time.time()))

import logging
from mercurial import ui, hg, localrepo
import nnet_train, sys, os
import htkdb_cm, shutil
import argparse
from helpers import compute_acc
import run_params_pb2
from google.protobuf.text_format import Merge

parser = argparse.ArgumentParser()
parser.add_argument('--skip_repo', dest='skip_repo', 
                    action='store_true', default=False, 
                    help='Do not check repository state')
parser.add_argument('--no_learn', dest='no_learn', 
                    action='store_true', default=False, 
                    help='Just do fwd prop on dev set')
parser.add_argument('--skip_notes', dest='skip_notes', 
                    action='store_true', default=False, 
                    help='Do not ask for input notes')
parser.add_argument('--num_files_per_load', type=int, default=-1, 
         help='Number of files loaded to gpu each time (-1 for all)')
parser.add_argument('--skip_borders', type=int, default=0, 
         help='Number of border frames to skip')
parser.add_argument('--nnet', type=str, default=None,
   help="Location of folder with NNet model to be used for initializing")
parser.add_argument('nn_def_file', help='Path to nn definition file')
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
input_model = os.path.join(arguments.nnet, "model.dat")
param_def_file = os.path.join(arguments.nnet, "run_params.txt")

output_file = os.path.join(arguments.output_fldr, "model.dat")
shutil.copy2(arguments.nn_def_file,
             os.path.join(arguments.output_fldr, "nn_def.txt"))
shutil.copy2(param_def_file,
             os.path.join(arguments.output_fldr, "run_params.txt"))

run_params = run_params_pb2.params()
f = open(param_def_file)
cfg_str = f.read()
f.close()
Merge(cfg_str, run_params)


data_src = htkdb_cm.htkdb_cm(arguments.db_name, arguments.db_path, 
                        run_params.data_params.num_frames_per_pt,
                        run_params.data_params.use_delta, 
                        skip_borders=arguments.skip_borders)
validation_src = htkdb_cm.htkdb_cm(arguments.validation_db_name,
                        arguments.db_path,
                        run_params.data_params.num_frames_per_pt,
                        run_params.data_params.use_delta)

nn_train = nnet_train.nn()
nn_train.create_nnet_from_def(arguments.nn_def_file, 
                              data_dim = data_src.get_data_dim(),
                              target_dim = data_src.get_label_dim())
nn_train.load(input_model)
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
err_tol = 0.25
eps_ratio=1.0
for i in range(run_params.max_epochs):
    if not arguments.no_learn:
        nn_train.train_one_epoch(data_src, eps_ratio)
    acc = compute_acc(nn_train, validation_src)
    err = 100-acc
    logging.info("Dev set FER = %.4f"%err)
    sys.stderr.write("Dev set FER = %.4f\n"%err)
    sys.stderr.flush()
    eps_ratio *= 0.5
    if err <= min_err:
        logging.info("Saving model to %s"%arguments.output_fldr)
        sys.stderr.write("Saving model to %s\n"%arguments.output_fldr)
        nn_train.save(output_file)
        min_err = err
    data_src.permute_file_indices_for_loading()

GPULock.FreeLock(id) 
