#! /usr/bin/python

# See README.txt for information and build instructions.

import run_params_pb2
import sys
from google.protobuf.text_format import MessageToString, Merge
import argparse

arg_parser = argparse.ArgumentParser("Create a definitions file for run_params")
arg_parser.add_argument('output_file', help='Path of output file')
arguments = arg_parser.parse_args()

run_def = run_params_pb2.params()
run_def.data_params.use_delta = True
run_def.data_params.normalization = run_params_pb2.DataParams.CMVN
run_def.data_params.num_frames_per_pt = 15

run_def.batch_size = 128
run_def.max_bad_epochs = 10
run_def.max_epochs = 40


# Write the new address book back to disk.
f = open(sys.argv[1], "wb")
f.write(MessageToString(run_def))
f.close()
