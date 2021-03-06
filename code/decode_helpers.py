from tempfile import NamedTemporaryFile
from pylab import *
import cudamat_ext as cm
import os, subprocess, sys
from  StripedFuncs import StripeData, UnStripeData
import logging
from scipy.signal import triang, hamming
 
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


def compute_predictions_for_sentence(db, nnet_model, fileNum, get_labels=False):
    data, labels = db.get_data_for_file(fileNum, return_labels=True)
    data_striped = StripeData(data, db.get_num_frames_per_pt(), 
                                                        append=True)
    dataDim, numFrames = data_striped.shape
    predictions = nnet_model.predict(data_striped, unnormalized=True)

    pred_class = predictions.argmax(axis=0)
    predictions = predictions - predictions.max(axis=0).reshape(1,-1, 
                                                                  order='F')
    #exp_predictions = exp(predictions)
    #exp_predictions = exp_predictions / exp_predictions.sum(axis=0).reshape(1,-1,order='F')
    #log_probs = log(exp_predictions+1e-20)

    predictions = predictions - log(exp(predictions).sum(axis=0).reshape(1,-1,order='F'))

    ones_matrix = eye(predictions.shape[0])
    class_matrix = ones_matrix[:, labels]
    log_probs = sum(predictions*class_matrix)

    num_correct = sum(pred_class == labels.reshape(-1))

    if get_labels:
       return predictions, num_correct, log_probs, pred_class, labels.reshape(-1)
    else:
       return predictions, num_correct, log_probs


def compute_predictions_for_sentence_multi(db, nnet_model, fileNum, 
                                           use_sum = False, get_labels=False,
                                           decoding_context=-1):
    data, labels = db.get_data_for_file(fileNum, return_labels=True)
    data_striped = StripeData(data, db.get_num_frames_per_pt(), 
                                                        append=True)
    dataDim, numFrames = data_striped.shape

    predictions = nnet_model.predict(data_striped, unnormalized=not use_sum)

    num_out_frames_per_pt = db.get_num_outputs_frames_per_pt()
    #if use_sum:
        #wts = hamming(num_out_frames_per_pt)[newaxis,:]
        #wts = tile(wts, (predictions.shape[0]/num_out_frames_per_pt,1)).reshape(-1,1, order='F')
        #prob = 0.1
        #predictions *= (prob*wts + (1-prob))
    
    if db.get_num_outputs_frames_per_pt() != 1 and decoding_context == -1:
        predictions = UnStripeData(predictions, db.get_num_outputs_frames_per_pt())
        extra_left = floor((db.get_num_outputs_frames_per_pt()-1)/2)
        extra_right = db.get_num_outputs_frames_per_pt()-1-extra_left
        predictions = predictions[:,extra_left:-extra_right]
    else:
        frame_dim = predictions.shape[0]/db.get_num_outputs_frames_per_pt()
        mid_frame=floor(db.get_num_outputs_frames_per_pt()/2)
        start_frame = mid_frame-decoding_context
        end_frame = mid_frame+decoding_context+1
        predictions = predictions[(frame_dim*(start_frame)):(frame_dim*(end_frame)),:]
        if decoding_context != 0:
            predictions = UnStripeData(predictions, decoding_context*2+1)
            predictions = predictions[:,decoding_context:-decoding_context]
        
    pred_class = predictions.argmax(axis=0)

    if not use_sum:
        cm_pred = cm.CUDAMatrix(predictions)
        cm_probs = cm.empty(cm_pred.shape).assign(0)
        cm_pred.compute_softmax(cm_probs)
        cm.log(cm_probs)
        predictions = cm_probs.asarray().copy()

        cm_probs.free_device_memory()
        cm_pred.free_device_memory()
        cm_probs, cm_pred = None, None
    else:
        predictions = log(predictions + 1e-35)

    ones_matrix = eye(predictions.shape[0])
    class_matrix = ones_matrix[:, labels]
    log_probs = sum(predictions*class_matrix)

    num_correct = sum(pred_class == labels.reshape(-1))

    if get_labels:
       return predictions, num_correct, log_probs, pred_class, labels.reshape(-1)
    else:
       return predictions, num_correct, log_probs


def compute_confusion_matrix(nnet_model, data_src):
    num_files = data_src.get_num_files()
    label_dim = data_src.get_label_dim()
    conf_matrix = zeros((label_dim, label_dim))
    printStr = ""
    tot_correct = 0
    tot_log_prob = 0
    num_pts = 0
    for fileNum in range(0, num_files): 
        predictions, num_correct, log_probs, pred_class, labels = \
          compute_predictions_for_sentence(data_src, nnet_model, fileNum, True)
        tot_log_prob += log_probs
        tot_correct += num_correct
        num_pts += predictions.shape[1]
        for (corr, pred) in zip(labels, pred_class): conf_matrix[corr, pred] += 1

        printStrNew = '\b' * (len(printStr)+1)
        printStr = "File # : %d, Accuraccy %.4f : %d of %d. lg(p) =%.3f"%(\
                       fileNum, tot_correct*100./num_pts, tot_correct, 
                       num_pts, tot_log_prob/num_pts)
        printString = printStrNew + printStr
        print printString,
        sys.stdout.flush()
 
    print "\nDone. Percent labels correct = ", str(tot_correct * 100./num_pts)
    print "Avg. log(prob) = ", str(tot_log_prob /num_pts)
    logging.info("Done. Percent labels correct = %.4f"%(tot_correct * 100./num_pts))
    logging.info("Avg. log(prob) = %.4f"%(tot_log_prob /num_pts))

    return conf_matrix



def write_kaldi_scores_file_old(utteranceIds, scores):
    f = NamedTemporaryFile(delete=False)
    numUtterances = len(utteranceIds)
    for utt_num in range(numUtterances):
       f.file.write(utteranceIds[utt_num])
       f.file.write(" [ ")
       for frame in range(scores[utt_num].shape[0]):
          scores[utt_num][frame,:].tofile(f.file, ' ', '%.6f')
          f.file.write("\n")
       f.file.write("]\n")
    f.close()
    f_name = f.name
    f = None
    return f_name
 
def write_kaldi_scores_file(utteranceIds, scores, f_name):
    f = open(f_name, 'w')
    numUtterances = len(utteranceIds)
    for utt_num in range(numUtterances):
       f.write(utteranceIds[utt_num])
       f.write(" [ ")
       for frame in range(scores[utt_num].shape[0]):
          scores[utt_num][frame,:].tofile(f, ' ', '%.6f')
          f.write("\n")
       f.write("]\n")
    f.close()
    return f_name
 
def create_predictions_file(nnet_model, data_src, score_file, 
                      priors_file = None, priors_scale=1.0, num_utt=-1):
                      
    try:
        priors2 = genfromtxt(priors_file)
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

    num_files = data_src.get_num_files()
    if num_utt != -1: num_files = min(num_utt, num_files)
    printStr = ""
    pred_lst =  []
    utt_id_lst = []
    tot_correct = 0
    tot_log_prob = 0
    num_pts = 0
    for fileNum in range(0, num_files): 
        predictions, num_correct, log_probs = \
                     compute_predictions_for_sentence(
                          data_src, nnet_model, fileNum)
        tot_log_prob += log_probs
        tot_correct += num_correct
        num_pts += predictions.shape[1]
        predictions = predictions - priors_scale*lgPriors
        pred_lst.append(predictions.transpose().copy())
        utt_id_lst.append(data_src._data_src.UtteranceIds[fileNum])

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

    return write_kaldi_scores_file(utt_id_lst, pred_lst, score_file)


def create_predictions_file_multi(nnet_model, data_src, score_file, use_sum,
                      priors_file = None, priors_scale=1.0, num_utt=-1, 
                      decoding_context=-1):
                      
    try:
        priors2 = genfromtxt(priors_file)
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

    num_files = data_src.get_num_files()
    if num_utt != -1: num_files = min(num_utt, num_files)
    printStr = ""
    pred_lst =  []
    utt_id_lst = []
    tot_correct = 0
    tot_log_prob = 0
    num_pts = 0
    for fileNum in range(0, num_files): 
        predictions, num_correct, log_probs = \
                     compute_predictions_for_sentence_multi(
                          data_src, nnet_model, fileNum, use_sum=use_sum, 
                          decoding_context=decoding_context)
        tot_log_prob += log_probs
        tot_correct += num_correct
        num_pts += predictions.shape[1]
        predictions = predictions - priors_scale*lgPriors
        pred_lst.append(predictions.transpose().copy())
        utt_id_lst.append(data_src._data_src.UtteranceIds[fileNum])

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

    return write_kaldi_scores_file(utt_id_lst, pred_lst, score_file)


def decode(nn, data_src, db_name, output_fldr, priors_file=None, priors_scale=1.0,
           wsj = False, num_utt=-1, decoding_context=-1):
    score_file = os.path.join(output_fldr, "scores.txt")
    create_predictions_file(nn, data_src, score_file, priors_file, priors_scale, num_utt)

    if wsj:
        #args = ["decode_kaldi_wsj.sh", score_file, output_fldr, db_name]
        print "Run: decode_kaldi_wsj.sh %s %s %s"%(score_file, output_fldr, db_name)
        return 0
    else:
        args = ["decode_nn_predictions.sh", score_file, output_fldr, 
             db_name]
    #os.remove(score_file)
    results = subprocess.check_output(args)
    parts = [float(x) for x in results.split()]
    per = parts[0]
    return per 

def decode_multi(nn, data_src, db_name, output_fldr, use_sum, 
                 priors_file=None, priors_scale=1.0, wsj = False, 
                 num_utt=-1, decoding_context=-1):
    score_file = os.path.join(output_fldr, "scores.txt")
    create_predictions_file_multi(nn, data_src, score_file, use_sum,
                                  priors_file, priors_scale, num_utt, decoding_context=decoding_context)

    if wsj:
        #args = ["decode_kaldi_wsj.sh", score_file, output_fldr, db_name]
        print "Run: decode_kaldi_wsj.sh %s %s %s"%(score_file, output_fldr, db_name)
        return 0
    else:
        args = ["decode_nn_predictions.sh", score_file, output_fldr, 
             db_name]
    #os.remove(score_file)
    results = subprocess.check_output(args)
    parts = [float(x) for x in results.split()]
    per = parts[0]
    return per 
