#!/usr/bin/env bash

. ./path.sh
. ./cmd.sh

nj=50
stage=0
stop_stage=200
data_conf=conf/data.conf # Location of the Kaldi generated data directory
feat_conf=conf/feat_melspec.conf
append_name=
feat_binary=make_melspectrogram_feats.sh  # make_modulation_feats.sh

. parse_options.sh || exit 1;

. $data_conf || exit 1;
. $feat_conf || exit 1;

if [ ! -z ${append_name} ] ; then
  append_name=`basename ${feat_conf} | cut -d '.' -f1`
fi

fbankdir=feats_dump/fbank_${append_name}
out_dir=exp/MI_WSJ_${append_name}
mkdir -p ${out_dir}
mkdir -p ${fbankdir}

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then

    echo "stage 1: Feature Generation"
    ${feat_binary} --cmd "$train_cmd" --nj $nj \
      --conf_file ${feat_conf} \
        ${data_dir} ${fbankdir} || exit 1;
    utils/fix_data_dir.sh ${data_dir} || exit 1;
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then

    echo "stage 1: Compute Mutual Information"
    compute_MI.sh --cmd "$train_cmd" --nj $nj \
      $fbankdir \
      ${ali_dir} \
      ${out_dir} || exit 1;
fi