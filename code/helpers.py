from numpy import sum
from StripedFuncs import *
import cudamat_ext as cm
import pdb

def compute_predictions_for_sentence(db, nnet_model, fileNum):
    data, labels = db.get_data_for_file(fileNum, return_labels=True)
    data_striped = StripeData(data, db.get_num_frames_per_pt(), 
                                                        append=True)
    dataDim, numFrames = data_striped.shape
    predictions = nnet_model.predict(data_striped, unnormalized=True)

    correct_class = predictions.argmax(axis=0)
    num_correct = sum(correct_class == labels.reshape(-1))
    return predictions, num_correct


def compute_acc(nnet_model, data_src):
    numDevFiles = data_src.get_num_files()
    printStr = ""
    devpreds =  []
    devUttIds = []
    tot_correct = 0
    num_pts = 0
    for fileNum in range(0, numDevFiles): 
        predictions, num_correct = compute_predictions_for_sentence(\
                                      data_src, nnet_model, fileNum)
        tot_correct += num_correct
        num_pts += predictions.shape[1]
    acc = tot_correct * 100./num_pts
    return  acc


def compute_predictions_for_sentence_mult(db, nnet_model, fileNum):
    data, labels = db.get_data_for_file(fileNum, return_labels=True)
    data_striped = StripeData(data, db.get_num_frames_per_pt(), 
                                                        append=True)
    dataDim, numFrames = data_striped.shape
    predictions = nnet_model.predict(data_striped, unnormalized=True)
    if db.get_num_outputs_frames_per_pt() != 1:
        predictions = UnStripeData(predictions, db.get_num_outputs_frames_per_pt())
        extra_left = floor((db.get_num_outputs_frames_per_pt()-1)/2)
        extra_right = db.get_num_outputs_frames_per_pt()-1-extra_left
        predictions = predictions[:,extra_left:-extra_right]

    correct_class = predictions.argmax(axis=0)
    num_correct = sum(correct_class == labels.reshape(-1))
    return predictions, num_correct

def compute_acc_mult(nnet_model, data_src):
    numDevFiles = data_src.get_num_files()
    printStr = ""
    devpreds =  []
    devUttIds = []
    tot_correct = 0
    num_pts = 0
    for fileNum in range(0, numDevFiles): 
        predictions, num_correct = compute_predictions_for_sentence_mult(\
                                      data_src, nnet_model, fileNum)
        tot_correct += num_correct
        num_pts += predictions.shape[1]
    acc = tot_correct * 100./num_pts
    return  acc


