from pylab import *
import numpy
numpy.random.seed(42) # This seed is meaningful. :).
import GPULock
GPULock.GetGPULock() 
import cudamat_ext as cm
cm.cublas_init()
cm.CUDAMatrix.init_random(42)

import logging
from mercurial import ui, hg, localrepo
import nnet_train, sys, os
import htkdb_cm, shutil
import argparse, time
from helpers import compute_acc

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument('--use_delta', action='store_true',
                        help="Use deltas and accelerations in training")
group.add_argument('--no_delta', action='store_true',
                        help="Dont use deltas and accelerations in training")
parser.add_argument('--skip_repo', dest='skip_repo', action='store_true',
                        default=False, help='Do not check repository state')
parser.add_argument('nn_def_file', help='Path to nn definition file')
parser.add_argument('--max_bad_epochs', type=int, default=8, 
                    help='maximum # of epochs allowed with increasing error')
parser.add_argument('--batch_size', type=int, default=128, 
                    help='Number of points in mini-batch')
parser.add_argument('--num_frames_per_pt', type=int, default=15, 
                             help='Number of frames per point')
parser.add_argument('--num_files_per_load', type=int, default=-1, 
                   help='Number of files loaded to gpu each time (-1 for all)')
parser.add_argument('db_name', help='Name of training database')
parser.add_argument('validation_db_name', help='Name of validation database')
parser.add_argument('db_path', help='Path to database')
parser.add_argument('max_epochs', type=int, 
                     help='Maximum number of epochs of training')
group = parser.add_mutually_exclusive_group()
group.add_argument("-cmn", "--speaker_cmn", action="store_true")
group.add_argument("-cmvn", "--speaker_cmvn", action="store_true")
group.add_argument("-n", "--normalize", action="store_true")
group.add_argument("-no_norm", "--no_norm", action="store_true")
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
logging.info(" ".join(sys.argv))

logging.info("Revision number for code: " + str(revision_num))
output_file = os.path.join(arguments.output_fldr, "model.dat")
shutil.copy2(arguments.nn_def_file,
             os.path.join(arguments.output_fldr, "nn_def.txt"))

if not arguments.normalize:
    logging.info("Not normalizing inputs.")

data_src = htkdb_cm.htkdb_cm(arguments.db_name, 
                             arguments.db_path, 
                             arguments.num_frames_per_pt,
                             arguments.use_delta)
validation_src = htkdb_cm.htkdb_cm(arguments.validation_db_name,
                                   arguments.db_path,
                                   arguments.num_frames_per_pt,
                                   arguments.use_delta)

nn_train = nnet_train.nn()
nn_train.create_nnet_from_def(arguments.nn_def_file, 
                              data_dim = data_src.get_data_dim(),
                              target_dim = data_src.get_label_dim())
nn_train.create_activations_and_probs(arguments.batch_size)
min_err = Inf
num_attempts = 0

data_src.setup_data(arguments.num_files_per_load,
                      arguments.speaker_cmn, 
                        arguments.speaker_cmvn,
                          arguments.normalize)
validation_src.setup_data(arguments.num_files_per_load,
                            arguments.speaker_cmn, 
                              arguments.speaker_cmvn,
                                arguments.normalize)
eps_ratio=1.0
for i in range(arguments.max_epochs):
    nn_train.train_one_epoch(data_src, eps_ratio)
    acc = compute_acc(nn_train, validation_src)
    err = 100-acc
    logging.info("Dev set FER = %.4f"%err)
    if err <= min_err:
        nn_train.save(output_file)
        min_err = err
    else:
        #eps_ratio *= 0.5
        num_attempts += 1

    if num_attempts >= arguments.max_bad_epochs:
        logging.info("Stopping criterion reached")
        break

    data_src.permute_indices_for_loaded_data()
