#!/usr/bin/env python3
# encoding: utf-8

"""
Author: samiksadhu, Johns Hopkins University
"""

'Generate a min-max file for binning histogram'

import argparse
import numpy as np
import bisect
import os
import kaldi_io
import pickle as pkl


def get_phoneme_labels(ali_dir):
    ali_files = []
    all_ali_dirs = ali_dir.split(',')
    for ali_dir in all_ali_dirs:
        ali_files.extend([os.path.join(ali_dir, f) for f in os.listdir(ali_dir) if f.startswith('ali.')])

    pdf_ali_dict = {}

    for file in ali_files:
        pdf_ali_file = "ark:ali-to-phones --per-frame {} ark:'gunzip -c {} |' ark:- |".format(
            os.path.join(ali_dir, "final.mdl"),
            file)
        pdf_ali_dict.update({u: d for u, d in kaldi_io.read_vec_int_ark(pdf_ali_file)})

    return pdf_ali_dict


def get_feats(feat_scp):
    return {uttid: feats for uttid, feats in kaldi_io.read_mat_scp(feat_scp)}


def get_minmax(dict_or_scp, scp_input=False, make_absolute=False):
    feat_min = +np.inf
    feat_max = -np.inf
    if scp_input:
        for _, feats in kaldi_io.read_mat_scp(dict_or_scp):
            if make_absolute:
                feats = np.abs(feats)
            one_max = np.max(feats)
            one_min = np.min(feats)
            if one_max > feat_max:
                feat_max = one_max
            if one_min < feat_min:
                feat_min = one_min
    else:
        for key in dict_or_scp:
            if make_absolute:
                dict_or_scp[key] = np.abs(dict_or_scp[key])
            one_max = np.max(dict_or_scp[key])
            one_min = np.min(dict_or_scp[key])
            if one_max > feat_max:
                feat_max = one_max
            if one_min < feat_min:
                feat_min = one_min

    return feat_min, feat_max


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Compute min-max values of labels and data for binning Histograms')
    parser.add_argument('scp', help='Feature scp file')
    parser.add_argument('phoneme_ali_dir', help='Phoneme alignment directory')
    parser.add_argument('out_file', help='Output file')
    parser.add_argument("--feat_size", type=int, default=80, help="Feature size")
    parser.add_argument("--make_absolute", type=bool, default=False,
                        help="Compute np.abs() on features before computing MI")
    args = parser.parse_args()

    all_alis = get_phoneme_labels(args.phoneme_ali_dir)

    mn_a, mx_a = get_minmax(dict_or_scp=all_alis)
    mn_f, mx_f = get_minmax(dict_or_scp=args.scp, scp_input=True, make_absolute=args.make_absolute)

    pkl.dump({'min': mn_a, 'max': mx_a}, open(args.out_file + '.ali.mnx', 'wb'))
    pkl.dump({'min': mn_f, 'max': mx_f}, open(args.out_file + '.feat.mnx', 'wb'))
