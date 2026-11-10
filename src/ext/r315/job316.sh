#!/bin/bash
# (316) зерно в склі: bash job316.sh GR [JIT] — L96 T300
R=/mnt/project-files/model/s52/r315; J=${2:-0.565}; d=gr$1_j$J; mkdir -p $R/$d $R/ck/$d; cd $R/$d
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 $R/run316.py jit=$J gr=$1 Tt=300 ck=10 ckdir=$R/ck/$d >> log 2>&1
