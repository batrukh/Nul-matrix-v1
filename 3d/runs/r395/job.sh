#!/bin/bash
# (395) bash job.sh SEED — тор без меж, α0,2, s0 3,45
R=/mnt/project-files/model/s52; d=$R/r395/runs/s345_seed$1; mkdir -p $d; cd $d; [ -f done ] && exit 0
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python3 -B $R/r396/rq383g.py sgn=0 L=16 s0=3.45 seed=$1 Tt=300 ell=4 ck=5 ac=0.2 >> log 2>&1 && touch done
