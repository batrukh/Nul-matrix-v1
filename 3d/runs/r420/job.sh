#!/bin/bash
# (420) bash job.sh SEED ; продовження з ck (rq420 бере лічильник кроків із *_ck.npz)
R=/mnt/project-files/model/s52; d=$R/r420/runs/M5_seed$1; [ -f $d/done ] && exit 0; mkdir -p $d; cd $d
CHAOS=M R=5 COMP=1 EARLY=0.8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python3 -B $R/r420/rq420.py sgn=0 L=16 s0=3.0 seed=$1 Tt=300 ell=4 ck=5 ac=0.1 >> log 2>&1 && touch done
