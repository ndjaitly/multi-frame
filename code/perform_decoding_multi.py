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
import nnet_train_multi_soft , sys, os
import htkdb_mult_cm, shutil
import argparse, time
from decode_helpers import *

parser = argparse.ArgumentParser()
parser.add_argument('--skip_repo', dest='skip_repo', action='store_true',
                        default=False, help='Do not check repository state')
parser.add_argument('--wsj', action='store_true', default=False, 
                     help='Are we decoding wsj')
parser.add_argument('--use_sum', action='store_true', default=False, 
                     help='Use Sum probabilites instead of geometric avg')
parser.add_argument('--decoding_context', type=int, default=-1, 
                     help='# of frames of context for decoding')
group = parser.add_mutually_exclusive_group()
group.add_argument('--use_delta', action='store_true',
                        help="Use deltas and accelerations in training")
group.add_argument('--no_delta', action='store_true',
                        help="Dont use deltas and accelerations in training")
parser.add_argument('--num_frames_per_pt', type=int, default=15, 
                             help='Number of frames per point')
parser.add_argument('--num_output_frames_per_pt', type=int, default=3, 
                             help='Number of target frames per point')
parser.add_argument('--num_utterances', type=int, default=-1, 
                             help='Number of utterances to write out')
parser.add_argument('db_name', help='Name of training database')
parser.add_argument('db_path', help='Path to database')
group = parser.add_mutually_exclusive_group()
group.add_argument("-cmn", "--speaker_cmn", action="store_true")
group.add_argument("-cmvn", "--speaker_cmvn", action="store_true")
group.add_argument("-n", "--normalize", action="store_true")
group.add_argument("-no_norm", "--no_norm", action="store_true")
parser.add_argument("-priors", default="", help="path to priors file")
parser.add_argument("-priors_scale", type=float, default=1.0, help="scaling for prior")
parser.add_argument('model_fldr', help='my neural network file')
parser.add_argument('run_name', help='name of run')
arguments = parser.parse_args()

model_file = os.path.join(arguments.model_fldr, "model.dat")
output_fldr = os.path.join(arguments.model_fldr, arguments.db_name, arguments.run_name)

if not arguments.skip_repo:
    rep = localrepo.instance(ui.ui(), '../', False)
    if sum([len(x) for x in rep.status()]) != 0:
        print "Please commit changes to repository before running program"
        sys.exit(1)


if not os.path.exists(output_fldr):
    os.makedirs(output_fldr)
logPath = os.path.join(output_fldr, "log.txt")
if os.path.exists(logPath): os.remove(logPath)
rep = localrepo.instance(ui.ui(), '../', False)
revision_num = rep.changelog.headrevs()[0] 

# create logger
logging.basicConfig(filename=logPath, level=logging.INFO, 
             format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.info("python " + " ".join(sys.argv))

logging.info("Revision number for code: " + str(revision_num))

if not arguments.normalize:
    logging.info("Not normalizing inputs.")

data_src = htkdb_mult_cm.htkdb_cm(arguments.db_name, 
                             arguments.db_path, 
                             arguments.num_frames_per_pt,
                             arguments.num_output_frames_per_pt,
                             arguments.use_delta)
data_src._speaker_cmn = arguments.speaker_cmn
data_src._speaker_cmvn = arguments.speaker_cmvn
data_src._normalize = arguments.normalize

nn_train = nnet_train_multi_soft .nn()
nn_train.load(model_file)
per = decode_multi(nn_train, data_src, arguments.db_name, output_fldr, 
                   arguments.use_sum, priors_file=arguments.priors, 
                   priors_scale=arguments.priors_scale,
                   wsj=arguments.wsj, num_utt=arguments.num_utterances,
                   decoding_context=arguments.decoding_context)

sys.stderr.write("PER = %.2f\n"%per)
sys.stderr.flush()
logging.info("PER = %.2f\n"%per)
