#!/bin/bash
# (425) bash job.sh ARM SEED  (M: α0,25 h0,36 — основна; S: α0,25 h0,39 — плече). Код (424), T600.
R=/mnt/project-files/model/s52; d=$R/r425/runs/$1_seed$2; mkdir -p $d; cd $d; [ -f done ] && exit 0
if [ $1 = M ]; then H=0.36; else H=0.39; fi
EARLY=0.7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python3 -B $R/r396/rq383g.py sgn=0 L=16 s0=3.0 seed=$2 Tt=600 ell=4 ck=5 ac=0.25 h=$H >> log 2>&1 && touch done
