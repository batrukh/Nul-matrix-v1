#!/bin/bash
# (316в): bash job316c.sh JIT [T=150] — скло зі стрілами з незсунутих точок (tb=1)
R=/mnt/project-files/model/s52/r315; T=${2:-150}; d=tb_j$1; mkdir -p $R/$d $R/ck/$d; cd $R/$d
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 $R/run316.py jit=$1 tb=1 Tt=$T ck=10 ckdir=$R/ck/$d >> log 2>&1
