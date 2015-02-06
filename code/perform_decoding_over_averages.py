from pylab import *
import numpy, subprocess
numpy.random.seed(42) # This seed is meaningful. :).
import GPULock
GPULock.GetGPULock() 
import cudamat_ext as cm
cm.cublas_init()
cm.CUDAMatrix.init_random(42)

import logging, run_params_pb2
from mercurial import ui, hg, localrepo
import nnet_train, sys, os
import htkdb_cm, shutil
import argparse, time
from decode_helpers import *
from google.protobuf.text_format import Merge

parser = argparse.ArgumentParser()
parser.add_argument('--skip_repo', dest='skip_repo', action='store_true',
                        default=False, help='Do not check repository state')
parser.add_argument('--wsj', action='store_true', default=False, 
                     help='Are we decoding wsj')
parser.add_argument('--use_sum', action='store_true', default=False, 
                     help='Use sum of probabilities rather than products')
parser.add_argument("-priors", default="", help="path to priors file")
parser.add_argument("-priors_scale", type=float, default=1.0, help="scaling for prior")
parser.add_argument('db_name', help='Name of training database')
parser.add_argument('db_path', help='Path to database')
parser.add_argument('model_list', help='File with list of neural network model folders')
parser.add_argument('output_fldr', help='Where to place results')
arguments = parser.parse_args()

if not arguments.skip_repo:
    rep = localrepo.instance(ui.ui(), '../', False)
    if sum([len(x) for x in rep.status()]) != 0:
        print "Please commit changes to repository before running program"
        sys.exit(1)


if not os.path.exists(arguments.output_fldr):
    os.makedirs(arguments.output_fldr)
logPath = os.path.join(arguments.output_fldr, "log.txt")
score_file = os.path.join(arguments.output_fldr, "scores.txt")
if os.path.exists(logPath): os.remove(logPath)
rep = localrepo.instance(ui.ui(), '../', False)
revision_num = rep.changelog.headrevs()[0] 


# create logger
logging.basicConfig(filename=logPath, level=logging.INFO, 
             format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.info("python " + " ".join(sys.argv))
logging.info("Revision number for code: " + str(revision_num))


f = open(arguments.model_list, 'r')
lines = f.readlines()
f.close()

try:
    priors2 = genfromtxt(arguments.priors)
    num_states = max(priors2[:,0])+1
    # give a minimum value of 1.
    priors = ones(num_states, dtype='float32')
    for i in range(priors2.shape[0]):
        priors[priors2[i,0]] = priors2[i,1]

    priors = priors*1.0/sum(priors)
    lgPriors = log(priors).reshape(-1,1)
    priors2 = None
    print "Priors loaded. First fiew lg priors = ", lgPriors[:10,0]
except Exception:
    print "No priors loaded"
    lgPriors = 0


pred_lst =  []
utt_id_lst = []
# Go through each model and collect results
for model_num, model_fldr in enumerate(lines):
    model_fldr = model_fldr.rstrip()
    # load run params
    run_params_file = os.path.join(model_fldr, "run_params.txt")
    run_params = run_params_pb2.params()
    f = open(run_params_file)
    cfg_str = f.read()
    f.close()
    Merge(cfg_str, run_params)

    data_src = htkdb_cm.htkdb_cm(arguments.db_name, 
                        arguments.db_path, 
                        run_params.data_params.num_frames_per_pt,
                        run_params.data_params.use_delta)

    cmn = run_params.data_params.normalization == run_params_pb2.DataParams.CMN
    cmvn = run_params.data_params.normalization == run_params_pb2.DataParams.CMVN
    normalize = run_params.data_params.normalization == run_params_pb2.DataParams.GLOB_NORMALIZE
    data_src.setup_data(1, cmn, cmvn, normalize)
    
    # load neural net. 
    model_file = os.path.join(model_fldr, "model.dat")
    nn_train = nnet_train.nn()
    nn_train.load(model_file)

    num_files = data_src.get_num_files()
    printStr = ""
    tot_correct = 0
    tot_log_prob = 0
    num_pts = 0

    for fileNum in range(0, num_files): 
        predictions, num_correct, log_probs = \
                     compute_predictions_for_sentence(
                          data_src, nn_train, fileNum)
        tot_log_prob += log_probs
        tot_correct += num_correct
        num_pts += predictions.shape[1]
        if model_num == len(lines)-1:
            predictions = predictions - arguments.priors_scale*lgPriors

        if model_num == 0:
            if arguments.use_sum:
                pred_lst.append(exp(predictions.transpose().copy()))
            else:
                pred_lst.append(predictions.transpose().copy())
            utt_id_lst.append(data_src._data_src.UtteranceIds[fileNum])
        else:
            if arguments.use_sum:
                pred_lst[fileNum] += exp(predictions.transpose())
            else:
                pred_lst[fileNum] += predictions.transpose()
            assert(utt_id_lst[fileNum] == data_src._data_src.UtteranceIds[fileNum])

        if model_num == len(lines)-1:
            pred_lst[fileNum] *= 1./len(lines)
            if arguments.use_sum:
                pred_lst[fileNum] = log(pred_lst[fileNum]+1e-35)

        printStrNew = '\b' * (len(printStr)+1)
        printStr = "File # : %d, Accuraccy %.4f : %d of %d. lg(p) =%.3f"%(\
                       fileNum, tot_correct*100./num_pts, tot_correct, 
                       num_pts, tot_log_prob/num_pts)
        printString = printStrNew + printStr
        print printString,
        sys.stdout.flush()
 
    print "Done. Percent labels correct = ", str(tot_correct * 100./num_pts)
    print "Avg. log(prob) = ", str(tot_log_prob /num_pts)
    logging.info("Done. Percent labels correct = %.4f"%(tot_correct * 100./num_pts))
    logging.info("Avg. log(prob) = %.4f"%(tot_log_prob /num_pts))


write_kaldi_scores_file(utt_id_lst, pred_lst, score_file)
if arguments.wsj:
    print "Run: decode_kaldi_wsj.sh %s %s %s"%(score_file, arguments.output_fldr, 
                                               arguments.db_name)
else:
    args = ["decode_nn_predictions.sh", score_file, arguments.output_fldr, \
            arguments.db_name]
    #os.remove(score_file)
    results = subprocess.check_output(args)
    logging.info("Results " + results)
    parts = [float(x) for x in results.split()]
    per = parts[0]
    sys.stderr.write("PER = %.2f\n"%per)
    sys.stderr.flush()
    logging.info("PER = %.2f\n"%per)
