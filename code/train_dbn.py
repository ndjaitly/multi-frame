import numpy, time
#numpy.random.seed(42) # This seed is meaningful. :).
numpy.random.seed(int(time.time())) # This seed is meaningful. :).
import GPULock
GPULock.GetGPULock() 
import cudamat_ext as cm
cm.cublas_init()
#cm.CUDAMatrix.init_random(42)
cm.CUDAMatrix.init_random(int(time.time()))
import sys
sys.path.insert(0, 'dbn')
import GaussianBinaryRBM, BinaryBinaryRBM
import dbn
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
parser.add_argument('--num_files_per_load', type=int, default=-1, 
         help='Number of files loaded to gpu each time (-1 for all)')
parser.add_argument('dbn_def_file', help='Path to nn definition file')
parser.add_argument('param_def_file', help='Path to run params file')
parser.add_argument('db_name', help='Name of training database')
parser.add_argument('dev_db_name', help='Name of validation database')
parser.add_argument('db_path', help='Path to database')
parser.add_argument('output_fldr', type=str, help='output folder')

arguments = parser.parse_args()
if not arguments.skip_repo:
    rep = localrepo.instance(ui.ui(), '.', False)
    if sum([len(x) for x in rep.status()]) != 0:
        print "Please commit changes to repository before running program"
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

notes_line = raw_input("Please enter notes for this experiment " + \
                  " (blank line indices finished):")
if not notes_line: logging.info("NOTES: <None provided>")
while (notes_line != ""):
    logging.info("NOTES: " + notes_line)
    notes_line = raw_input()


logging.info("Revision number for code: " + str(revision_num))
output_file = os.path.join(arguments.output_fldr, "model.dat")
shutil.copy2(arguments.dbn_def_file,
             os.path.join(arguments.output_fldr, "dbn_def.txt"))
shutil.copy2(arguments.param_def_file,
             os.path.join(arguments.output_fldr, "run_params.txt"))

run_params = run_params_pb2.params()
f = open(arguments.param_def_file)
cfg_str = f.read()
f.close()
Merge(cfg_str, run_params)

# No vtlp used in DBN training.
data_src = htkdb_cm.htkdb_cm(arguments.db_name, arguments.db_path, 
                        run_params.data_params.num_frames_per_pt,
                        run_params.data_params.use_delta)
dev_src = htkdb_cm.htkdb_cm(arguments.dev_db_name, arguments.db_path, 
                        run_params.data_params.num_frames_per_pt,
                        run_params.data_params.use_delta)

cmn = run_params.data_params.normalization == run_params_pb2.DataParams.CMN
cmvn = run_params.data_params.normalization == run_params_pb2.DataParams.CMVN
normalize = run_params.data_params.normalization == run_params_pb2.DataParams.GLOB_NORMALIZE
data_src.setup_data(arguments.num_files_per_load, cmn, cmvn, normalize)
dev_src.setup_data(arguments.num_files_per_load, cmn, cmvn, normalize)

dbn_train = dbn.dbn(arguments.dbn_def_file)
dbn_train.train(data_src, dev_src, run_params.batch_size, arguments.output_fldr, 
                    reload_if_exists=True)
