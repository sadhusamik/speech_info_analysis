#!/usr/bin/env bash

. ./path.sh

# Mel Spectrum options

nj=100
cmd=queue.pl
add_reverb=large_room
overlap_fraction=0.5
fduration=1.5
speech_type='clean'
srate=16000
exec_file='acc_log_spectrum.py'

. parse_options.sh || exit 1;

data_dir=$1
feat_dir=$2

echo "$0 $@"

# Convert feat_dir to the absolute file name

feat_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir";} print $dir; ' $feat_dir ${PWD}`

mkdir -p $feat_dir

name=`basename $data_dir`

add_segment=false
if [ -f "$data_dir/segments" ]; then
    echo "Adding segment files from $data_dir/segments"
    add_segment=true
fi
log_dir=$data_dir/log

mkdir -p $log_dir
log_dir=`realpath ${log_dir}`


# split data dir
echo "$0: Splitting data dir"
split_data.sh ${data_dir} $nj || exit 1;


echo "$0: Computing average spectral features for scp files..."

# Compute mel spectrum features

if $add_segment; then
  $cmd --mem 5G JOB=1:$nj \
  $log_dir/acc_spectrum_${name}_${speech_type}_${add_reverb}.JOB.log \
  ${exec_file}  \
    $data_dir/split${nj}/JOB/wav.scp \
    $feat_dir/avg_spectrum_${name}.JOB.pkl \
    --segment_file=$data_dir/split${nj}/JOB/segments \
    --add_reverb=${add_reverb} \
    --fduration=${fduration} \
    --overlap_fraction=${overlap_fraction} \
    --speech_type=${speech_type} \
    --srate=16000 || exit 1;
else
  $cmd --mem 5G JOB=1:$nj \
    $log_dir/acc_spectrum_${name}_${speech_type}_${add_reverb}.JOB.log \
    ${exec_file} \
      $data_dir/split${nj}/JOB/wav.scp \
      $feat_dir/avg_spectrum_${name}.JOB.pkl \
      --add_reverb=${add_reverb} \
      --fduration=${fduration} \
      --overlap_fraction=${overlap_fraction} \
      --speech_type=${speech_type} \
      --srate=16000 || exit 1;
fi

echo $0": Finished computing average spectrum for $name"
