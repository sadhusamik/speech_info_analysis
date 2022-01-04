#!/usr/bin/env bash

. ./path.sh

nj=20
cmd="queue.pl"
analyze_transitions=false
only_combine=false
shifts='0'
conf_file=

. parse_options.sh || exit 1;

data_dir=$1
ali_dir=$2
out_dir=$3

if [ ! -z ${conf_file} ] ; then
  source ${conf_file}
fi

name=`basename $data_dir`

for file in `ls ${data_dir}/*.scp`; do
  cat ${file} || exit 1;
done > $data_dir/all_feats

scp=$data_dir/all_feats
log_dir=$out_dir/log
mkdir -p $log_dir
mkdir -p $out_dir
log_dir=`realpath ${log_dir}`
feat_size=`feat-to-dim scp:$scp -`
add_opts=""
if $analyze_transitions; then 
  add_opts="$add_opts --analyze_transitions"
fi

if ! $only_combine; then
  ## Divide the data and compute MI for each part
  echo "$0: Computing min-max of all features"
  echo "$0: Log file can be found in $log_dir/getminmax.*.log "
  $cmd --mem 2G JOB=1 \
    $log_dir/getminmax.JOB.log \
     compute_minmax.py \
     $scp \
     $ali_dir \
     $out_dir/minmax \
     --feat_size=$feat_size || exit 1 ;

  split_scp=""
  for n in $(seq $nj); do
    split_scp="$split_scp $log_dir/feats.$n.scp"
  done

  utils/split_scp.pl $scp $split_scp || exit 1;

  echo "$0: Computing MI"
  echo "$0: Log file can be found in $log_dir/compute_MI.*.log"

  $cmd --mem 10G JOB=1:$nj \
    $log_dir/compute_MI.JOB.log \
    compute_signal_label_confusion_matrix.py \
      $log_dir/feats.JOB.scp \
      $ali_dir \
      $out_dir/minmax.ali.mnx \
      $out_dir/minmax.feat.mnx \
      $out_dir/MI_${name}.JOB $add_opts \
      --feat_size=$feat_size \
      --shifts=$shifts|| exit 1 ;
fi

# Combine all the MI data

$cmd --mem 2G JOB=1 \
  $log_dir/combine_histogram_dumps.JOB.log \
  combine_histogram_dumps.py \
    $out_dir || exit 1;
