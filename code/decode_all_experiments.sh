#!/bin/bash
if [ $# -ne 1 ]; then
   echo "Usage: $0 [# of layers]"
   exit 1
fi 

layer=$1

for db in dev test
do 
   for run in 1 2 3
   do 
       python perform_decoding_multi.py --use_sum --use_delta -n ${db} --num_output_frames=15 /ais/gobi3/u/ndjaitly/workspace/Thesis/final/alignments/TIMIT/FBANKS_E/  \
        /ais/gobi3/u/ndjaitly/workspace/Thesis/final/prod/nnets/${layer}layer_multisoft_15_slower_yet_${run} ${db}_sum
 
      sleep 5
   done
done
