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
    parser = argparse.ArgumentParser('Extract Modulation Features (FDLP-spectrogram OR M-vectors)')
    parser.add_argument('scp', help='scp file')
    parser.add_argument('outfile', help='output file')
    parser.add_argument("--segment_file", default=None, type=str, help="segment file will be used if provided")
    parser.add_argument('--fduration', type=float, default=0.02, help='Window length (0.02 sec)')
    parser.add_argument('--overlap_fraction', type=float, default=0.15, help='Overlap fraction for overlap-add')
    parser.add_argument('--srate', type=int, default=16000, help='Sampling rate of the signal')
    parser.add_argument('--append_len', type=int, default=1000000, help='Append zeros to make signal this long')
    parser.add_argument('--add_reverb', default=None, help='location of the RIR file')

    return parser.parse_args()


def compute_modulations(args):
    # Define FDLP class
    feat_model = FDLP(fduration=args.fduration, overlap_fraction=args.overlap_fraction, srate=args.srate)

    acc_logmag = np.zeros(args.append_len)
    acc_phase = np.zeros(args.append_len)
    count = 0

    # Load reverberation files if provided
    add_reverb = args.add_reverb
    if add_reverb is not None:
        sr_r, rir = read(add_reverb)
        rir = rir[:, 0]
        rir = rir / np.power(2, 15)
    else:
        print('%s: No reverberation added!' % sys.argv[0])

    # Feature extraction
    if args.segment_file is None:
        with ReadHelper('scp:' + args.scp) as reader:
            for key, (rate, signal) in reader:

                print('%s: Computing Features for file: %s' % (sys.argv[0], key))
                sys.stdout.flush()

                # add reverberation
                if add_reverb is not None:
                    signal, idx_shift = addReverb_nodistortion(signal, rir)
                    # if not add_reverb == 'clean':

                    #    if args.speech_type == 'clean':
                    # signal = np.concatenate([np.zeros(idx_shift), signal])
                    #        signal = np.concatenate([signal, np.zeros(signal_rev.shape[0] - signal.shape[0])])
                    #        sig_out = signal
                    #    elif args.speech_type == 'reverb':
                    #        sig_out = signal_rev
                    #    else:
                    #        raise ValueError("speech_type can only be 'clean' or 'reverb'")
                    # else:
                    #    sig_out = signal
                #signal = signal[0:16000 * 4]
                cc, logmag, phase = feat_model.acc_log_spectrum_fft(signal, append_len=args.append_len, discont=3*np.pi/2)
                if cc is not None:
                    acc_logmag += logmag
                    acc_phase += phase
                    count += cc
    else:
        with ReadHelper('scp:' + args.scp, segments=args.segment_file) as reader:
            for key, (rate, signal) in reader:

                print('%s: Computing Features for file: %s' % (sys.argv[0], key))
                sys.stdout.flush()

                # add reverberation
                if add_reverb is not None:
                    signal, idx_shift = addReverb_nodistortion(signal, rir)

                # if add_reverb:
                #    if not add_reverb == 'clean':
                #        signal_rev, idx_shift = addReverb_nodistortion(signal, rir)
                #        if args.speech_type == 'clean':
                #            # signal = np.concatenate([np.zeros(idx_shift), signal])
                #            signal = np.concatenate([signal, np.zeros(signal_rev.shape[0] - signal.shape[0])])
                #            sig_out = signal
                #        elif args.speech_type == 'reverb':
                #            sig_out = signal_rev
                #        else:
                #            raise ValueError("speech_type can only be 'clean' or 'reverb'")
                #    else:
                #        sig_out = signal

                cc, logmag, phase = feat_model.acc_log_spectrum_fft(signal, append_len=args.append_len)
                if cc is not None:
                    acc_logmag += logmag
                    acc_phase += phase
                    count += cc

    pkl.dump({'count': count, 'acc_logmag': acc_logmag, 'acc_phase': acc_phase}, open(args.outfile, 'wb'))


if __name__ == '__main__':
    args = get_args()
    compute_modulations(args)
