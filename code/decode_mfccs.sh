#!/bin/bash
if [ $# -ne 1 ] ; then
  echo Usage: $0 [run_name]
  exit 1
fi

run_name=$1
#for num_layers in 2 3 4 5 ; do
for num_layers in 6 7 ; do
  cmd="python perform_decoding.py --use_delta -n test /ais/gobi3/u/ndjaitly/workspace/Thesis/final/alignments/TIMIT/MFCC /ais/gobi3/u/ndjaitly/workspace/Thesis/final/kaldi_train/runs/mfcc_${num_layers}layer_${run_name} test"
   echo $cmd
   $cmd
   sleep 2
done
