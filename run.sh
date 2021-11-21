#!/usr/bin/env bash

. ./path.sh
. ./cmd.sh

nj=30
stage=0
stop_stage=200
data_conf=conf/data.conf # Location of the Kaldi geenrated data directory
feat_conf=conf/feat_melspec.conf

. parse_options.sh || exit 1;

. $data_conf || exit 1;
. $feat_conf || exit 1;

fbankdir=fbank_${nfilters}_${nfft}_frate${frate}

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then

    echo "stage 1: Feature Generation"
    make_melspectrogram_feats.sh --cmd "$train_cmd" --nj $nj \
      --conf_file ${feat_conf} \
        ${data_dir} ${fbankdir}
    utils/fix_data_dir.sh ${data_dir}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then

    echo "stage 1: Compute Mutual Information"
    compute_MI.sh --cmd "$train_cmd" --nj $nj \
      --conf ${data_conf} || exit 1;
fi