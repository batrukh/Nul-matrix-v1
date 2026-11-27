#!/bin/bash
# (425В) скан h біля переходу: bash job2.sh H SEED  (α0,25, код (424)/(425), T600, EARLY 0,7)
R=/mnt/project-files/model/s52; d=$R/r425/runs/h$1_seed$2; mkdir -p $d; cd $d; [ -f done ] && exit 0
EARLY=0.7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python3 -B $R/r396/rq383g.py sgn=0 L=16 s0=3.0 seed=$2 Tt=600 ell=4 ck=5 ac=0.25 h=$1 >> log 2>&1 && touch done
