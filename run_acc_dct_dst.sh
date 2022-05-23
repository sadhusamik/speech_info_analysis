#!/usr/bin/env bash

. ./path.sh
. ./cmd.sh

nj=50
stage=0

data_conf=conf/data.conf # Location of the Kaldi generated data directory
add_reverb='large_room'
overlap_fraction=0.5
fduration=1.5


append_name=
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
    ${feat_binary} --cmd "$train_cmd" --nj $nj \
      --add_reverb 'clean' \
      --overlap_fraction ${overlap_fraction} \
      --fduration ${fduration} \
        ${data_dir} ${fbankdir} || exit 1;
    #utils/fix_data_dir.sh ${data_dir} || exit 1;
fi
