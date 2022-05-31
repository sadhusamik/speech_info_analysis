#!/usr/bin/env bash

. ./path.sh
. ./cmd.sh

nj=50
stage=0

data_conf=conf/data_small.conf # Location of the Kaldi generated data directory
add_reverb='large_room'
overlap_fraction=0.75
fduration=5
speech_type='clean'

append_name=${speech_type}_${add_reverb}
feat_binary=compute_average_spectrum.sh  # make_modulation_feats.sh

. parse_options.sh || exit 1;

. $data_conf || exit 1;

if [ -z ${append_name} ] ; then
  append_name=`basename ${feat_conf} | cut -d '.' -f1`
fi

fbankdir=feats_dump_dst_dct/fbank_${append_name}
mkdir -p ${fbankdir}

if [ ${stage} -le 0 ] ; then

    echo "stage 1: Feature Generation"
    bash ${feat_binary} --cmd "$train_cmd" --nj $nj \
      --add_reverb ${add_reverb} \
      --overlap_fraction ${overlap_fraction} \
      --fduration ${fduration} \
      --speech_type ${speech_type} \
      ${data_dir} ${fbankdir} || exit 1;
    #utils/fix_data_dir.sh ${data_dir} || exit 1;
fi
