#!/bin/bash
if [ $# -ne 1 ]; then
   echo "Usage: $0 [# of layers]"
   exit 1
fi 

layer=$1

for run in 1 2 3
do 
    python train_nnet_mult.py --skip_notes --dbn /ais/gobi3/u/ndjaitly/workspace/Thesis/final/kaldi_train/dbns/timit/timit_2K_7Layers_1/ --num_files_per_load=1000 params/timit_${layer}layers_multisoft_raw_slower2.txt params/bs_128_nf_15_norm_mult_15.txt train dev /ais/gobi3/u/ndjaitly/workspace/Thesis/final/alignments/TIMIT/FBANKS_E/ /ais/gobi3/u/ndjaitly/workspace/Thesis/final/prod/nnets/${layer}layer_multisoft_15_slower_yet_${run}
    sleep 5
done
