#!/bin/bash
# (374б) bash jobL.sh SEED — L24, α 0,1, T100
R=/mnt/project-files/model/s52; d=$R/r374/L24_seed$1; mkdir -p $d; cd $d; [ -f done ] && exit 0
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python3 -B $R/r351/rq351f.py L=24 s0=3.0 seed=$1 Tt=100 ell=4 ck=5 ac=0.1 >> log 2>&1 && touch done
