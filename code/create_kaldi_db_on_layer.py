import numpy, shutil
from numpy import zeros, argsort, arange, log, exp, concatenate, sqrt, dot, tile
import nnet_train_multi_soft
from mercurial import ui, hg, localrepo
import sys, os
import GPULock, util, HTK , htkdb_cm
id = GPULock.GetGPULock() 
import cudamat_ext as cm
numpy.random.seed(42) # This seed is meaningful. :).
cm.cublas_init()
cm.CUDAMatrix.init_random(42)

import logging, StripedFuncs
import argparse, time
from google.protobuf.text_format import Merge
import run_params_pb2


parser = argparse.ArgumentParser()
parser.add_argument('--skip_repo', dest='skip_repo', 
                    action='store_true', default=False, 
                    help='Do not check repository state')
parser.add_argument('--probabilities', dest='probabilities', 
           action='store_true', default=False, 
           help='Output probabilities rather than activations')
parser.add_argument('db_name', help='Path to database')
parser.add_argument('output_path', help='Path to raw file')
parser.add_argument('in_db_path', help='Path to database')
parser.add_argument('model_fldr', type=str, help='folder with model')
parser.add_argument('output_layer', type=int, 
                     help='0-based index to the layer we want outputs from')

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

db_path = os.path.join(arguments.output_path, arguments.db_name)
if not os.path.exists(db_path):
    os.makedirs(db_path)
logPath = os.path.join(db_path, "log.txt")
if os.path.exists(logPath): os.remove(logPath)
rep = localrepo.instance(ui.ui(), '.', False)
revision_num = rep.changelog.headrevs()[0] 

# create logger
logging.basicConfig(filename=logPath, level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.info("python " + " ".join(sys.argv))

logging.info("Revision number for code: " + str(revision_num))
# create logger
notes_line = raw_input("Please enter notes for this db creation " + \
                  " (blank line indices finished):")
if not notes_line: 
    logging.info("NOTES: <None provided>")
while (notes_line != ""):
    logging.info("NOTES: " + notes_line)
    notes_line = raw_input()


run_params_file = os.path.join(arguments.model_fldr, "run_params.txt")
run_params = run_params_pb2.params()
f = open(run_params_file)
cfg_str = f.read()
f.close()
Merge(cfg_str, run_params)

numFramesPerDataPoint = run_params.data_params.num_frames_per_pt


data_src = htkdb_cm.htkdb_cm(arguments.db_name, arguments.in_db_path, 
                     numFramesPerDataPoint,
                     use_deltas_accs=run_params.data_params.use_delta)
cmn = run_params.data_params.normalization == \
                             run_params_pb2.DataParams.CMN
cmvn = run_params.data_params.normalization == \
                             run_params_pb2.DataParams.CMVN
normalize = run_params.data_params.normalization == \
                   run_params_pb2.DataParams.GLOB_NORMALIZE
data_src.setup_data(1, cmn, cmvn, normalize)

nn_def_file = os.path.join(arguments.model_fldr, "nn_def.txt")
model_file = os.path.join(arguments.model_fldr, "model.dat")

cp_nn_def_file = os.path.join(db_path, "nn_def.txt")
shutil.copy2(nn_def_file, cp_nn_def_file)
cp_run_params = os.path.join(db_path, "run_params.txt")
shutil.copy2(run_params_file, cp_run_params)
cp_model_file = os.path.join(db_path, "model.dat")
shutil.copy2(model_file, cp_model_file)

nnet = nnet_train_multi_soft.nn()
nnet.load(model_file)


num_files = data_src.get_num_files()
logging.info("# of files = %d\n"%(num_files))
num_pts = 0
raw_file_list = []
Utt2Speaker = data_src._data_src.Utt2Speaker
Speaker2Utt = data_src._data_src.Speaker2Utt

for file_num in range(num_files):
    if file_num % 10 == 0:
        sys.stderr.write('.')
        sys.stderr.flush()
    if file_num % 800 == 0 and file_num != 0:
        sys.stderr.write('\n')
        sys.stderr.flush()
 
    inputs = data_src.get_data_for_file(file_num, return_labels=False)
    num_frames = inputs.shape[1]
    left = tile(inputs[:,0].reshape(-1,1), (1, numFramesPerDataPoint/2))
    right = tile(inputs[:,-1].reshape(-1,1), 
                  (1, int((numFramesPerDataPoint-1)/2)))
    inputs = concatenate((left, inputs, right), axis=1)
    inputs = StripedFuncs.StripeData(inputs, numFramesPerDataPoint)

    data = nnet.fwd_prop_np(inputs, not arguments.probabilities,
                                        last_layer=arguments.output_layer)

    assert(data.shape[1] == num_frames)
    utterance_id = data_src._data_src.UtteranceIds[file_num]
    raw_file_name = utterance_id + ".htk"

    raw_path = os.path.join(db_path, raw_file_name)
    raw_file_list.append(raw_file_name)
    HTK.WriteHTK(raw_path, data)

    try:
        sum_data += data.sum(axis=1)
        sum_data_sq += (data**2).sum(axis=1)
        num_pts += data.shape[1]

        speaker = Utt2Speaker[utterance_id]
        SpeakerMeans[speaker] += data.sum(axis=1).reshape(-1,1)
        SpeakerStds[speaker] += (data**2).sum(axis=1).reshape(-1,1)
        SpeakerNumFrames[speaker] += data.shape[1]

    except NameError:
        sum_data = data.sum(axis=1)
        sum_data_sq = (data**2).sum(axis=1)
        num_pts = data.shape[1]

        print "Creating Speaker means and stdevs"

        SpeakerMeans = {}
        SpeakerStds = {}
        SpeakerNumFrames = {}
        if Speaker2Utt is None:
            raise Exception, "Input db needs to have attribute Speaker2Utt"

        for speaker in Speaker2Utt.keys():
            SpeakerMeans[speaker] = zeros((data.shape[0],1))
            SpeakerStds[speaker] = zeros((data.shape[0],1))
            SpeakerNumFrames[speaker] = zeros((data.shape[0],1))

        speaker = Utt2Speaker[utterance_id]
        SpeakerMeans[speaker] = data.sum(axis=1).reshape(-1,1)
        SpeakerStds[speaker] = (data**2).sum(axis=1).reshape(-1,1)
        SpeakerNumFrames[speaker] = data.shape[1]


for speaker in Speaker2Utt.keys():
    SpeakerMeans[speaker] /= (1.0 * SpeakerNumFrames[speaker])
    SpeakerStds[speaker] -= SpeakerNumFrames[speaker] * \
                            (SpeakerMeans[speaker]**2)
    SpeakerStds[speaker] /= (1.0 *SpeakerNumFrames[speaker]-1)
    SpeakerStds[speaker][SpeakerStds[speaker] < 1e-8] = 1e-8
    SpeakerStds[speaker] = sqrt(SpeakerStds[speaker])


data_mean = sum_data/num_pts

var = ((sum_data_sq - num_pts * data_mean**2)/num_pts)
var[var < 1e-8] = 1e-8
data_std = sqrt(var)

ali_file = os.path.join(db_path, "ali")
shutil.copy2(data_src._data_src.AliFile, ali_file)
db_file = os.path.join(db_path, "%s.dat"%arguments.db_name)
util.save(db_file, 'label_dim UtteranceIds RawFileList data_dim \
          DataMeanVect DataStdVect Utt2Speaker Speaker2Utt \
          SpeakerMeans SpeakerStds lst_ignored_files', 
         {'label_dim':data_src.get_label_dim(),
          'UtteranceIds':data_src._data_src.UtteranceIds,
          'RawFileList':raw_file_list, 
          'data_dim':3*data.shape[0],  # fake value of 3 for compliance with htkdb_cm
          'DataMeanVect':data_mean, 
          'DataStdVect':data_std, 
          'Utt2Speaker':Utt2Speaker,
          'Speaker2Utt':Speaker2Utt,
          'SpeakerMeans':SpeakerMeans,
          'SpeakerStds':SpeakerStds,
          'lst_ignored_files':[]})

logging.info("Done creating db")

GPULock.FreeLock(id) 
