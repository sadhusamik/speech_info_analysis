#!/usr/bin/env python3
# encoding: utf-8

"""
Author: samiksadhu, Johns Hopkins University
"""

import numpy as np
from utils import get_kaldi_ark, addReverb, addReverb_nodistortion
from scipy.io.wavfile import read
import subprocess
import argparse
import sys
import io
from fdlp.fdlp import FDLP
import pickle as pkl
import logging


def get_args():
    parser = argparse.ArgumentParser('Extract Modulation Features (FDLP-spectrogram OR M-vectors)')
    parser.add_argument('scp', help='scp file')
    parser.add_argument('outfile', help='output file')
    parser.add_argument("--scp_type", default='wav', help="scp type can be 'wav' or 'segment'")
    parser.add_argument('--fduration', type=float, default=0.02, help='Window length (0.02 sec)')
    parser.add_argument('--overlap_fraction', type=float, default=0.15, help='Overlap fraction for overlap-add')
    parser.add_argument('--srate', type=int, default=16000, help='Sampling rate of the signal')
    parser.add_argument('--append_zero_factor', type=int, default=100, help='expand signal with this factor of zeros')
    parser.add_argument('--add_reverb', help='input "clean" OR "small_room" OR "large_room"')
    parser.add_argument('--speech_type', default='clean', type=str, help="'clean' OR 'reverb'")

    return parser.parse_args()


def compute_modulations(args):
    # Define FDLP class
    feat_model = FDLP(fduration=args.fduration, overlap_fraction=args.overlap_fraction, srate=args.srate)

    N = int(args.fduration * args.srate)
    acc_dct = np.zeros(N * (args.append_zero_factor + 1))
    acc_dst = np.zeros(N * (args.append_zero_factor + 1))
    count = 0

    with open(args.scp, 'r') as fid:

        add_reverb = args.add_reverb
        if add_reverb:
            if add_reverb == 'small_room':
                sr_r, rir = read('./RIR/RIR_SmallRoom1_near_AnglA.wav')
                rir = rir[:, 1]
                rir = rir / np.power(2, 15)
            elif add_reverb == 'large_room':
                sr_r, rir = read('./RIR/RIR_LargeRoom1_far_AnglA.wav')
                rir = rir[:, 1]
                rir = rir / np.power(2, 15)
            elif add_reverb == 'clean':
                print('%s: No reverberation added!' % sys.argv[0])
            else:
                raise ValueError('Invalid type of reverberation!')

        for line in fid:

            tokens = line.strip().split()
            uttid, inwav = tokens[0], ' '.join(tokens[1:])

            print('%s: Computing Features for file: %s' % (sys.argv[0], uttid))
            sys.stdout.flush()

            if args.scp_type == 'wav':
                if inwav[-1] == '|':
                    try:
                        proc = subprocess.run(inwav[:-1], shell=True, stdout=subprocess.PIPE)
                        sr, signal = read(io.BytesIO(proc.stdout))
                        skip_rest = False
                    except Exception:
                        skip_rest = True
                else:
                    try:
                        sr, signal = read(inwav)
                        skip_rest = False
                    except Exception:
                        skip_rest = True

                assert sr == args.srate, 'Input file has different sampling rate.'
            elif args.scp_type == 'segment':
                try:
                    cmd = 'wav-copy ' + inwav + ' - '
                    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
                    sr, signal = read(io.BytesIO(proc.stdout))
                    skip_rest = False
                except Exception:
                    skip_rest = True
            else:
                raise ValueError('Invalid type of scp type, it should be either wav or segment')
            signal = signal / np.power(2, 15)

            if not skip_rest:
                if add_reverb:
                    if not add_reverb == 'clean':
                        signal_rev, idx_shift = addReverb_nodistortion(signal, rir)
                    if args.speech_type == 'clean':
                        signal = np.concatenate([np.zeros(idx_shift), signal])
                        signal = np.concatenate([signal, np.zeros(signal_rev.shape[0] - signal.shape[0])])
                        sig_out = signal
                    elif args.speech_type == 'reverb':
                        sig_out = signal_rev
                    else:
                        raise ValueError("speech_type can only be 'clean' or 'reverb'")

                cc, dct_sum, dst_sum = feat_model.acc_log_spectrum(sig_out[np.newaxis, :],
                                                                   append_zero_factor=args.append_zero_factor)
                acc_dct += dct_sum
                acc_dst += dst_sum
                count += cc

    pkl.dump({'count': count, 'acc_dct': acc_dct, 'acc_dst': acc_dst}, open(args.outfile, 'wb'))


if __name__ == '__main__':
    args = get_args()
    compute_modulations(args)
