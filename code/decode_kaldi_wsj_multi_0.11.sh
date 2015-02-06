#!/bin/bash
kaldi_dir=/u/ndjaitly/workspace/KALDI/kaldi-trunk
export PATH=${kaldi_dir}/src/bin:${kaldi_dir}/tools/openfst/bin:${kaldi_dir}/src/fstbin/:${kaldi_dir}/src/gmmbin/:${kaldi_dir}/src/featbin/:${kaldi_dir}/src/lm/:${kaldi_dir}/src/sgmmbin/:${kaldi_dir}/src/fgmmbin/:${kaldi_dir}/src/latbin/:$PWD:$PATH

export LC_ALL=C
if [ $# -ne 3 ] ; then
   echo Usage: $0 [scores file] [decoding folder] [db name]
   exit
fi 

cmd=local/run.pl

pred_file=$1
dir=$2
db_name=$3
mkdir -p $dir

kaldi_path=/u/ndjaitly/workspace/KALDI/kaldi-trunk/egs/wsj/s5/
data=${kaldi_path}/data/$db_name
model=${kaldi_path}/exp/tri4b/final.mdl
graphdir=${kaldi_path}/exp/tri4b/graph_bd_tgpr
#graphdir=${kaldi_path}/exp/tri4b/graph_tgpr
max_active=7000
# good combination
#beam=25
#latbeam=9
#acwt=0.08

#beam=30
#latbeam=10
#acwt=0.1

# multi predict
beam=25
latbeam=9
acwt=0.11

#acwt=0.04
min_lmwt=7
max_lmwt=15

cat $pred_file | latgen-faster-mapped --max-active=$max_active --beam=$beam --lattice-beam=$latbeam \
  --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt \
  $model $graphdir/HCLG.fst ark,t:- "ark:|gzip -c > $dir/lat.1.gz" || exit 1;


# Run the scoring
[ ! -x local/score.sh ] && \
  echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
local/score.sh --min-lmwt $min_lmwt --max-lmwt $max_lmwt --cmd "$cmd" $data $graphdir $dir 2>$dir/scoring.log || exit 1;

