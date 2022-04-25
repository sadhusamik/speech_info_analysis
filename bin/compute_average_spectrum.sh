#!/usr/bin/env bash

. ./path.sh

# Mel Spectrum options

nj=100
cmd=queue.pl
add_reverb=large_room
overlap_fraction=0.5
fduration=1.5
srate=16000

. parse_options.sh || exit 1;

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


# split files

echo $0": Splitting segment OR scp files for parallalization..."

echo "$0: Splitting scp files..."

split_scp=""
for n in $(seq $nj); do
split_scp="$split_scp $log_dir/wav_${name}.$n.scp"
done

utils/split_scp.pl $scp $split_scp || exit 1;

echo "$0: Computing Mel Spectral features for scp files..."

# Compute mel spectrum features

$cmd --mem 5G JOB=1:$nj \
  $log_dir/acc_spectrum_${name}.JOB.log \
  acc_log_spectrum.py \
    $log_dir/wav_${name}.JOB.scp \
    $feat_dir/avg_spectrum_${name}.JOB.pkl \
    --add_reverb=${add_reverb} \
    --fduration=${fduration} \
    --overlap_fraction=${overlap_fraction} \
    --srate=16000 || exit 1;

rm $log_dir/wav_${name}.*.scp


echo $0": Finished computing average spectrum for $name"
