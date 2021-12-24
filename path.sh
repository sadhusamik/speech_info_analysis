export LC_ALL=C

# Add util files
export PATH=$PWD/utils:$PATH
# Add bash binary files
export PATH=$PWD/bin:$PATH
# Add python binaries
export PATH=$PWD/src/feature_extraction:$PWD/src/info_theory:$PATH
# Add kaldi_io files to pythonpath
export PYTHONPATH=$PWD/tools/kaldi-io-for-python:$PWD/src/feature_extraction:$PWD/tools/fdlp_spectrogram:$PYTHONPATH



# Other pythonpath ..

export OMP_NUM_THREADS=1

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8