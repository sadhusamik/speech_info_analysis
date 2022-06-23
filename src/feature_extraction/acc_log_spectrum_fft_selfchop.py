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
from kaldiio import ReadHelper


def get_args():
    parser = argparse.ArgumentParser('Extract statistics with chopped speech signals')
    parser.add_argument('scp', help='scp file')
    parser.add_argument('outfile', help='output file')
    parser.add_argument('--fduration', type=float, default=10, help='Window length (10 seconds)')
    parser.add_argument('--overlap_fraction', type=float, default=0.5, help='Overlap fraction for overlap-add')
    parser.add_argument('--srate', type=int, default=16000, help='Sampling rate of the signal')
    parser.add_argument('--append_len', type=int, default=1000000, help='Append zeros to make signal this long')
    parser.add_argument('--add_reverb', help='input "clean" OR "small_room" OR "large_room"')
    parser.add_argument('--speech_type', default='clean', type=str, help="'clean' OR 'reverb'")

    return parser.parse_args()


def compute_modulations(args):
    # Define FDLP class
    feat_model = FDLP(fduration=args.fduration, overlap_fraction=args.overlap_fraction, srate=args.srate)

    acc_logmag = np.zeros(args.append_len)
    acc_phase = np.zeros(args.append_len)
    count = 0

    # Load reverberation files if provided
    add_reverb = args.add_reverb
    if add_reverb:
        if add_reverb == 'small_room':
            sr_r, rir = read('./RIR/RIR_SmallRoom1_near_AnglA.wav')
            rir = rir[:, 0]
            rir = rir / np.power(2, 15)
        elif add_reverb == 'large_room':
            sr_r, rir = read('./RIR/RIR_LargeRoom1_far_AnglA.wav')
            rir = rir[:, 0]
            rir = rir / np.power(2, 15)
        elif add_reverb == 'clean':
            print('%s: No reverberation added!' % sys.argv[0])
        else:
            raise ValueError('Invalid type of reverberation!   ')

    # Feature extraction
    idx = 0
    count=0
    frate = 1 / (args.fduration - args.overlap_fraction * args.fduration)
    flength_samples = int(args.srate * args.fduration)
    frate_samples = int(args.srate / frate)
    with ReadHelper('scp:' + args.scp) as reader:
        for key, (rate, signal_whole) in reader:

            while idx + frate_samples < signal_whole.shape[0]:
                signal = signal_whole[idx:idx + flength_samples]
                count+=1
                print('%s: Computing Features for file %s chunk number %d' % (sys.argv[0], key, count))
                sys.stdout.flush()
                idx += frate_samples
                # add reverberation
                if add_reverb:
                    if not add_reverb == 'clean':
                        signal_rev, idx_shift = addReverb_nodistortion(signal, rir)
                        if args.speech_type == 'clean':
                            # signal = np.concatenate([np.zeros(idx_shift), signal])
                            signal = np.concatenate([signal, np.zeros(signal_rev.shape[0] - signal.shape[0])])
                            sig_out = signal
                        elif args.speech_type == 'reverb':
                            sig_out = signal_rev
                        else:
                            raise ValueError("speech_type can only be 'clean' or 'reverb'")
                    else:
                        sig_out = signal

                cc, logmag, phase = feat_model.acc_log_spectrum_fft(sig_out, append_len=args.append_len)
                if cc is not None:
                    acc_logmag += logmag
                    acc_phase += phase
                    count += cc

    pkl.dump({'count': count, 'acc_logmag': acc_logmag, 'acc_phase': acc_phase}, open(args.outfile, 'wb'))


if __name__ == '__main__':
    args = get_args()
    compute_modulations(args)
