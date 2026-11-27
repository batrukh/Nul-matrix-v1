#!/bin/bash
# bash job449.sh SEED NS  (α0,25 h0,36 T600, як (425) M, інший шум)
R=/mnt/project-files/model/s52; d=$R/r449/runs/s$1_ns$2; mkdir -p $d; cd $d; [ -f done ] && exit 0
NS=$2 EARLY=0.7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python3 -B $R/r449/rq449.py sgn=0 L=16 s0=3.0 seed=$1 Tt=600 ell=4 ck=5 ac=0.25 h=0.36 >> log 2>&1 && touch done
