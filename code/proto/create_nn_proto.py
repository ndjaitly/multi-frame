#! /usr/bin/python

# See README.txt for information and build instructions.

import nn_pb2
import sys
from google.protobuf.text_format import MessageToString, Merge
import argparse

def prompt_for_layer(layer):
    layer.name = raw_input("Enter name: ")
    layer.input_dim = int(raw_input("# of inputs: "))
    layer.num_units = int(raw_input("# of  units: "))

    type = raw_input("Unit type:")
    if type.lower() == "linear":
        layer.type = nn_pb2.Layer.LINEAR
    elif type.lower() == "sigmoid":
        layer.type = nn_pb2.Layer.SIGMOID
    elif type.lower() == "softmax":
        layer.type = nn_pb2.Layer.SOFTMAX
    elif type.lower() == "relu":
        layer.type = nn_pb2.Layer.RELU
    else:
        print "Unknown unit type."

    prompt_for_init_params(layer.init_params)
    prompt_for_learning_schedule(layer.learning_schedule.add())
    while raw_input("Add another schedule piece (y/n)").lower() == "y":
        prompt_for_learning_schedule(layer.learning_schedule.add())
    


def prompt_for_init_params(init_params):
    init_params.wt_sigma = float(raw_input("weight_sigma: "))
    init_params.biases_min = float(raw_input("biases_min: "))
    init_params.biases_max = float(raw_input("biases_max: "))

def prompt_for_learning_schedule(schedule):
    schedule.l1_wt = float(raw_input("l1 wt: "))
    schedule.l2_wt = float(raw_input("l2 wt: "))
    schedule.epsilon = float(raw_input("epsilon: "))
    schedule.epsilon_anneal_rate = float(raw_input("epsilon anneal rate: "))
    schedule.momentum = float(raw_input("momentum: "))
    schedule.start_epoch = int(raw_input("start_epoch: "))
    schedule.end_epoch = int(raw_input("end_epoch: "))
    schedule.dropout_rate = float(raw_input("dropout_rate: "))
    schedule.wt_norm_constraint = float(raw_input("wt_norm_constraint: "))


arg_parser = argparse.ArgumentParser("Create a definitions file for nn")
arg_parser.add_argument('output_file', help='Path of output file')
arguments = arg_parser.parse_args()

nn_def = nn_pb2.nn()

while raw_input("Add Layer (y/n)").lower() == "y":
    prompt_for_layer(nn_def.layers.add())

# Write the new address book back to disk.
f = open(sys.argv[1], "wb")
f.write(MessageToString(nn_def))
f.close()
