#!/bin/bash

if [ $# != 4 -a $# != 3 ]; then 
   echo "Incorrect arguments. Usage: $0 [score_file] [output_folder] [db name] [acoustic_scale |optional]"
   exit 1;
fi

acousticscale=1.0
pred_file=$1
outdir=$2
db_name=$3
if [ $# == 4 ]; then
   acousticscale=$4
fi

kaldi_dir=/u/ndjaitly/workspace/KALDI/kaldi-trunk
export PATH=${kaldi_dir}/src/bin:${kaldi_dir}/tools/openfst/bin:${kaldi_dir}/src/fstbin/:${kaldi_dir}/src/gmmbin/:${kaldi_dir}/src/featbin/:${kaldi_dir}/src/lm/:${kaldi_dir}/src/sgmmbin/:${kaldi_dir}/src/fgmmbin/:${kaldi_dir}/src/latbin/:$PWD:$PATH
export LC_ALL=C
beam=22
kaldi_recipe_fldr=${kaldi_dir}/egs/timit/s5
symtab=${kaldi_recipe_fldr}/data/lang/words.txt
model=${kaldi_recipe_fldr}/exp/mono/final.mdl
graphdir=${kaldi_recipe_fldr}/exp/mono/graph

#outdir=$$.tmp
trans_file=$outdir/$db_name.tra
ali_file=$outdir/$db_name.ali

ref_file=${kaldi_recipe_fldr}/data/$db_name/text
TFILE="$outdir/$(basename $0).$$.tmp"

mapping="en:n,ao:aa,ax-h:ah,ax:ah,ix:ih,el:l,zh:sh,ux:uw,axr:er,em:m,nx:n,eng:ng,hv:hh,pcl:pau,tcl:pau,kcl:pau,q:pau,bcl:pau,dcl:pau,gcl:pau,epi:pau"
cat $pred_file | decode-faster-mapped --beam=$beam \
                      --acoustic-scale=$acousticscale \
                      --word-symbol-table=$symtab $model $graphdir/HCLG.fst \
                      ark,t:- ark,t:$trans_file ark,t:$ali_file 2>> $outdir/log

scripts/collapse_phones.pl  --ignore-first-field  $symtab \
                               $mapping < $trans_file > $TFILE
 
score=$(scripts/sym2int.pl --ignore-first-field $symtab $ref_file | \
      scripts/collapse_phones.pl --ignore-first-field $symtab $mapping |\
      compute-wer --mode=present ark:-  ark,p:$TFILE 2>/dev/null | grep "%WER" | awk '{print $2 " " $11; }')
echo $score 
rm $TFILE
rm $outdir/log
#echo TFILE: $TFILE
#echo  $outdir
#echo $outdir.log
