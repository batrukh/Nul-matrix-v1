#!/bin/bash
# (424) bash job.sh ARM SEED  (A: α0,1 h0,24; B: α0,25 h0,36)
R=/mnt/project-files/model/s52; d=$R/r424/runs/$1_seed$2; mkdir -p $d; cd $d; [ -f done ] && exit 0
if [ $1 = A ]; then AC=0.1; H=0.24; else AC=0.25; H=0.36; fi
EARLY=0.7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python3 -B $R/r396/rq383g.py sgn=0 L=16 s0=3.0 seed=$2 Tt=300 ell=4 ck=5 ac=$AC h=$H >> log 2>&1 && touch done
