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
    parser.add_argument('--use_frames', type=bool, default=False, help='Use frames instead whole utterance')
    parser.add_argument("--append_time", default=None, type=float,
                        help="Duration of speech to accumulate before processing")
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

    if args.append_time is not None:
        wavfile = np.zeros(1)
        time = 0
        time_limit = args.append_time
        time_limit = time_limit * 60
        county = 0

    # Feature extraction
    if args.segment_file is None:
        with ReadHelper('scp:' + args.scp) as reader:
            for key, (rate, signal) in reader:
                if args.append_time is not None:
                    if time <= time_limit:
                        wavfile = np.concatenate([wavfile, np.zeros(int(args.fduration * args.srate))])
                        wavfile = np.concatenate([wavfile, signal])
                        time += signal.shape[0] / args.srate
                    else:
                        county += 1
                        # add reverberation
                        if add_reverb is not None:
                            L = wavfile.shape[0]
                            wavfile, idx_shift = addReverb_nodistortion(wavfile, rir)
                            wavfile = wavfile[0:L]
                        print('%s: Computing Features appended speech file number: %d, duration: %f seconds' % (
                            sys.argv[0], county, wavfile.shape[0] / 16000))
                        sys.stdout.flush()
                        if args.use_frames:
                            cc, logmag, phase = feat_model.acc_log_spectrum_fft_frames(wavfile,
                                                                                       append_len=args.append_len,
                                                                                       discont=np.pi)
                        else:
                            cc, logmag, phase = feat_model.acc_log_spectrum_fft(wavfile, append_len=args.append_len,
                                                                                discont=np.pi)
                        if cc is not None:
                            acc_logmag += logmag
                            acc_phase += phase
                            count += cc

                        wavfile = np.zeros(1)
                        time = 0
                else:
                    print('%s: Computing Features for file: %s' % (sys.argv[0], key))
                    sys.stdout.flush()
                    # add reverberation
                    if add_reverb is not None:
                        L = signal.shape[0]
                        signal, idx_shift = addReverb_nodistortion(signal, rir)
                        signal = signal[0:L]

                    if args.use_frames:
                        cc, logmag, phase = feat_model.acc_log_spectrum_fft_frames(signal, append_len=args.append_len,
                                                                                   discont=np.pi)
                    else:
                        cc, logmag, phase = feat_model.acc_log_spectrum_fft(signal, append_len=args.append_len,
                                                                            discont=np.pi)

                    if cc is not None:
                        acc_logmag += logmag
                        acc_phase += phase
                        count += cc
    else:
        with ReadHelper('scp:' + args.scp, segments=args.segment_file) as reader:
            for key, (rate, signal) in reader:

                if args.append_time is not None:
                    if time <= time_limit:
                        wavfile = np.concatenate([wavfile, np.zeros(int(args.fduration * args.srate))])
                        wavfile = np.concatenate([wavfile, signal])
                        time += signal.shape[0] / args.srate
                    else:
                        county += 1
                        sys.stdout.flush()
                        # add reverberation
                        L = wavfile.shape[0]
                        if add_reverb is not None:
                            wavfile, idx_shift = addReverb_nodistortion(wavfile, rir)
                            wavfile = wavfile[0:L]
                        print('%s: Computing Features appended speech file number: %d, duration: %f seconds' % (
                            sys.argv[0], county, wavfile.shape[0] / 16000))
                        if args.use_frames:
                            cc, logmag, phase = feat_model.acc_log_spectrum_fft_frames(wavfile,
                                                                                       append_len=args.append_len,
                                                                                       discont=np.pi)
                        else:
                            cc, logmag, phase = feat_model.acc_log_spectrum_fft(wavfile, append_len=args.append_len,
                                                                                discont=np.pi)

                        if cc is not None:
                            acc_logmag += logmag
                            acc_phase += phase
                            count += cc
                        wavfile = np.zeros(1)
                        time = 0
                else:
                    print('%s: Computing Features for file: %s' % (sys.argv[0], key))
                    sys.stdout.flush()
                    # add reverberation
                    if add_reverb is not None:
                        L = signal.shape[0]
                        signal, idx_shift = addReverb_nodistortion(signal, rir)
                        signal = signal[0:L]

                    if args.use_frames:
                        cc, logmag, phase = feat_model.acc_log_spectrum_fft_frames(signal, append_len=args.append_len,
                                                                                   discont=np.pi)
                    else:
                        cc, logmag, phase = feat_model.acc_log_spectrum_fft(signal, append_len=args.append_len,
                                                                            discont=np.pi)

                    if cc is not None:
                        acc_logmag += logmag
                        acc_phase += phase
                        count += cc

    pkl.dump({'count': count, 'acc_logmag': acc_logmag, 'acc_phase': acc_phase}, open(args.outfile, 'wb'))


if __name__ == '__main__':
    args = get_args()
    compute_modulations(args)
