#!/usr/bin/env python3
# encoding: utf-8

"""
Author: samiksadhu, Johns Hopkins University
"""

'Computing Joint Histogram of features and Labels'

import argparse
import numpy as np
import bisect
import os
import kaldi_io
import pickle as pkl


def get_minmax(feat_dict):
    feat_min = +np.inf
    feat_max = -np.inf
    for key in feat_dict:
        one_max = np.max(feat_dict[key])
        one_min = np.min(feat_dict[key])
        if one_max > feat_max:
            feat_max = one_max
        if one_min < feat_min:
            feat_min = one_min

    return feat_min, feat_max


def get_signal_label_joint_distribution(alis, feat_scp, minmax_ali, minmax_feat, shifts, feat_dim=80, num_bins=100,
                                        make_absolute=False, frequency_scaling=None):
    shifts = [int(x) for x in shifts.split(',')]
    num_shifts = len(shifts)
    if frequency_scaling:
        frequency_scaling = [x for x in frequency_scaling.split(',')]
        frequency_scaling[0] = int(frequency_scaling[0])  # num_filters
        frequency_scaling[1] = int(frequency_scaling[1])  # num_freq
        frequency_scaling[2] = float(frequency_scaling[2])  # freq_resolution
        freq_multiplier = np.linspace(0, frequency_scaling[1] * frequency_scaling[2], frequency_scaling[1])

    mnx_a = pkl.load(open(minmax_ali, 'rb'))
    mnx_f = pkl.load(open(minmax_feat, 'rb'))
    mn_a, mx_a = mnx_a['min'], mnx_a['max']
    mn_f, mx_f = mnx_f['min'], mnx_f['max']
    sig_bins = np.linspace(mn_f, mx_f, num_bins + 1)
    dist = np.zeros((num_shifts, feat_dim, num_bins, mx_a))
    cc = 0
    for uttid, feats in kaldi_io.read_mat_scp(feat_scp):
        cc += 1
    nums = cc
    count = 0
    absent_keys = 0
    for key, feats in kaldi_io.read_mat_scp(feat_scp):
        count += 1
        if make_absolute:
            feats = np.abs(feats)
        if frequency_scaling:
            feats = np.reshape(feats, (-1, frequency_scaling[0], frequency_scaling[1]))
            feats *= freq_multiplier
            feats = np.reshape(feats, (-1, frequency_scaling[0] * frequency_scaling[1]))

        print('Processing {:f} % of files'.format(count * 100 / nums))
        if key in alis:
            for sh_idx, sh in enumerate(shifts):
                one_feat = np.roll(feats, shift=sh, axis=0)
                for idx, label in enumerate(alis[key]):
                    f = one_feat[idx, :]
                    for r in range(feat_dim):
                        ii = int(bisect.bisect_left(sig_bins, f[r]))
                        jj = label - 1
                        if ii == 0:
                            ii = 1
                        if ii == num_bins + 1:
                            ii = num_bins
                        ii = ii - 1
                        dist[sh_idx, r, ii, jj] += 1
        else:
            absent_keys += 1

    print('{:d}/{:d} number of keys were absent in the alignment dictionary'.format(absent_keys, nums))
    return dist


def get_signal_trans_joint_distribution(alis, feats, minmax_ali, minmax_feat, feat_dim=80, num_bins=100):
    mnx_f = pkl.load(open(minmax_feat, 'rb'))
    mn_f, mx_f = mnx_f['min'], mnx_f['max']
    sig_bins = np.linspace(mn_f, mx_f, num_bins + 1)
    dist = np.zeros((feat_dim, num_bins, 2))
    nums = len(list(feats.keys()))
    count = 0
    for key in feats:
        count += 1
        print('Processing {:f} % of files'.format(count * 100 / nums))
        for idx, label in enumerate(alis[key]):
            f = feats[key][idx, :]
            for r in range(feat_dim):
                ii = int(bisect.bisect_left(sig_bins, f[r]))
                jj = int(label)
                if ii == 0:
                    ii = 1
                if ii == num_bins + 1:
                    ii = num_bins
                ii = ii - 1
                dist[r, ii, jj] += 1

    return dist


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


def get_transitions(alis):
    trans_dict = {}
    for utt in alis:
        one_ali = alis[utt]
        r_prev = 2222222
        one_trans = np.zeros(one_ali.shape[0])
        for idx, r in enumerate(one_ali):
            if idx > 0:
                if r_prev != r:
                    # There is a transition here
                    one_trans[idx] = 1
                    one_trans[idx - 1] = 1
                    one_trans[idx + 1] = 1
            r_prev = r
        trans_dict[utt] = one_trans

    return trans_dict


def get_feats(feat_scp):
    return {uttid: feats for uttid, feats in kaldi_io.read_mat_scp(feat_scp)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Compute Signal-Label Confusion Matrix')
    parser.add_argument('scp', help='Feature scp file')
    parser.add_argument('phoneme_ali_dir', help='Phoneme alignment directory')
    parser.add_argument('minmax_ali', help='Alignmnet minmax file')
    parser.add_argument('minmax_feat', help='Feature minmax file')
    parser.add_argument('out_file', help='Output file')
    parser.add_argument("--feat_size", type=int, default=80, help="Feature size")
    parser.add_argument("--make_absolute", type=bool, default=False,
                        help="Compute np.abs() on features before computing MI")
    parser.add_argument("--analyze_transitions", action="store_true", help="Set to compute MI at transitions")
    parser.add_argument("--frequency_scaling", type=str, default=None,
                        help="If scaling by 1/f you can set this option as num_filters,num_freq_components,freq_resolution [Option used when computing MI of modulation spectrum]")
    parser.add_argument("--shifts", type=str, default='0',
                        help="Shift features along time axis along these dimension eg. '-1,0,1'")
    args = parser.parse_args()

    all_alis = get_phoneme_labels(args.phoneme_ali_dir)
    if args.analyze_transitions:
        all_alis = get_transitions(all_alis)

    if args.analyze_transitions:
        dist = get_signal_trans_joint_distribution(all_alis, args.scp, args.minmax_ali, args.minmax_feat,
                                                   feat_dim=args.feat_size, num_bins=100)
    else:
        dist = get_signal_label_joint_distribution(all_alis, args.scp, args.minmax_ali, args.minmax_feat, args.shifts,
                                                   feat_dim=args.feat_size, num_bins=100,
                                                   make_absolute=args.make_absolute,
                                                   frequency_scaling=args.frequency_scaling)
    pkl.dump(dist, open(args.out_file + '.pkl', 'wb'))
