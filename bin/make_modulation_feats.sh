#!/usr/bin/env bash

. ./path.sh

# Feature Options
nj=100
n_filters=80
fduration=1.5
frate=100
coeff_num=100
coeff_range='1,100'
order=100
overlap_fraction=0.15
lifter_file=
lfr=10
return_mvector=False
srate=16000
cmd=queue.pl
add_opts=

write_utt2num_frames=false
derivative_signal=false

conf_file=

. parse_options.sh || exit 1;

# Overwirte options from config file
if [ ! -z ${conf_file} ] ; then
  source ${conf_file}
fi

data_dir=$1
feat_dir=$2

echo "$0 $@"

# Convert feat_dir to the absolute file name

feat_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir";} print $dir; ' $feat_dir ${PWD}`

mkdir -p $feat_dir

name=`basename $data_dir`
scp=$data_dir/wav.scp
log_dir=$data_dir/log
mkdir -p $log_dir
log_dir=`realpath ${log_dir}`

if ${write_utt2num_frames}; then
    add_opts="$add_opts --write_utt2num_frames"
fi

if ${derivative_signal}; then
    echo "Using derivative signal"
    add_opts="$add_opts --derivative"
fi

if [ ! -z ${lifter_file} ] ; then
  add_opts="$add_opts --lifter_file=${lifter_file}"
fi

echo $0": Splitting scp files for parallalization..."

split_scp=""
for n in $(seq $nj); do
  split_scp="$split_scp $log_dir/wav_${name}.$n.scp"
done

utils/split_scp.pl $scp $split_scp || exit 1;

echo "$0: Computing modulation features for scp files..."

# Compute mel spectrum features

  $cmd --mem 5G JOB=1:$nj \
    $log_dir/feats_${name}.JOB.log \
    compute_modulation_features.py \
      $log_dir/wav_${name}.JOB.scp \
      $feat_dir/modspec_${name}.JOB \
      $add_opts \
      --n_filters=$n_filters \
      --fduration=$fduration \
      --frate=$frate \
      --coeff_num=${coeff_num} \
      --coeff_range=${coeff_range} \
      --order=$order \
      --overlap_fraction=${overlap_fraction} \
      --lfr=$lfr \
      --return_mvector=${return_mvector} \
      --srate=$srate || exit 1;

  # concatenate all scp files together

  for n in $(seq $nj); do
    cat $feat_dir/modspec_$name.$n.scp || exit 1;
  done > $data_dir/feats.scp

  rm $log_dir/wav_${name}.*.scp

  # concatenate all length files together
  if $write_utt2num_frames; then
      for n in $(seq $nj); do
        cat $feat_dir/modspec_$name.$n.len || exit 1;
      done > $data_dir/utt2num_frames
  fi


echo $0": Finished computing mel spectrum features for $name"
