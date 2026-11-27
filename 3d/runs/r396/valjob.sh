#!/bin/bash
# (396) перевірка фізики на швидкому коді: α0,1 T100, як (373); bash valjob.sh SEED
R=/mnt/project-files/model/s52; d=$R/r396/val/a0.1_seed$1; mkdir -p $d; cd $d; [ -f done ] && exit 0
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python3 -B $R/r396/rq383g.py sgn=0 L=16 s0=3.0 seed=$1 Tt=100 ell=4 ck=5 ac=0.1 >> log 2>&1 && touch done
