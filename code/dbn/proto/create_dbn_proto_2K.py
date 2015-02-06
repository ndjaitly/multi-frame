#! /usr/bin/python

# See README.txt for information and build instructions.

import dbn_pb2
import sys
from google.protobuf.text_format import MessageToString, Merge
import argparse


def add_layer_config(config, input_dim, num_units, layer_name, 
                                           layer_type):
    config.name = layer_name
    config.input_dim = input_dim
    config.num_units = num_units

    config.type = layer_type

    config.wt_sigma = 0.01
    config.vis_bias = 0
    config.hid_bias = -1

    config.epsilon_w = 1e-4
    config.epsilon_b = 1e-4
    config.initial_momentum = 0.8
    config.final_momentum = 0.9
    config.mom_switch_epoch = 3
    config.l2_decay = 0
    config.num_epochs = 30


arg_parser = argparse.ArgumentParser("Create a definitions file for dbn")
arg_parser.add_argument('num_hid', type=str, 
                           help='# of units e.g. 2000:2000')
arg_parser.add_argument('output_file', help='Path of output file')
arguments = arg_parser.parse_args()

lst_num_units = [int(x) for x in arguments.num_hid.split(":")]
num_layers = len(lst_num_units)
dbn_def = dbn_pb2.dbn()
add_layer_config(dbn_def.layer_configs.add(), -1, lst_num_units[0], 
                             "Layer0", dbn_pb2.LayerConfig.GAUSSIAN_BINARY)
for layer_num in range(num_layers-1):
    add_layer_config(dbn_def.layer_configs.add(), lst_num_units[layer_num],
                  lst_num_units[layer_num+1], "Layer" + str(layer_num+1), 
                      dbn_pb2.LayerConfig.BINARY_BINARY)

# Write the new address book back to disk.
f = open(arguments.output_file, "wb")
f.write(MessageToString(dbn_def))
f.close()
