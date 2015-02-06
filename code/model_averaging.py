from StripedFuncs import *
import os, subprocess
def compute_predictions_for_sentence(db, nnet_model, fileNum, num_averages = 10, get_labels=False):
    data, labels = db.get_data_for_file(fileNum, return_labels=True)
    data_striped = StripeData(data, db.get_num_frames_per_pt(), 
                                                        append=True)
    dataDim, numFrames = data_striped.shape
    for  i in range(num_averages):
        predictions = nnet_model.predict(data_striped, unnormalized=False)
        if i == 0:   
            predictions_sum = predictions.copy()
        else:
            predictions_sum += predictions
      
    predictions = log(predictions_sum*1./num_averages + 1e-35)
    pred_class = predictions.argmax(axis=0)

    ones_matrix = eye(predictions.shape[0])
    class_matrix = ones_matrix[:, labels]
    log_probs = sum(predictions*class_matrix)

    num_correct = sum(pred_class == labels.reshape(-1))

    return predictions, num_correct, log_probs


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
 

def perform_averaged_decoding(nnet_model, data_src, num_averages, 
                              output_fldr, db_name):
    score_file = os.path.join(output_fldr, "scores.txt")
    num_files = data_src.get_num_files()
    printStr = ""
    pred_lst =  []
    utt_id_lst = []
    tot_correct = 0
    tot_log_prob = 0
    num_pts = 0
    for fileNum in range(0, num_files): 
        predictions, num_correct, log_probs = \
                     compute_predictions_for_sentence( data_src, 
                  nnet_model, fileNum, num_averages=num_averages)
        tot_log_prob += log_probs
        tot_correct += num_correct
        num_pts += predictions.shape[1]
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
    write_kaldi_scores_file(utt_id_lst, pred_lst, score_file)

    args = ["decode_nn_predictions.sh", score_file, output_fldr, 
            db_name]
    results = subprocess.check_output(args)
    print results
    parts = [float(x) for x in results.split()]
    per = parts[0]
    return per 
