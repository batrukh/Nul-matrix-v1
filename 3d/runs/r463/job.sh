#!/bin/bash
R=/mnt/project-files/model/s52; d=$R/r463/runs/$1_seed$2; mkdir -p $d; cd $d; [ -f done ] && exit 0
case $1 in N0) H=0.0; LC=6.5;; N15) H=0.15; LC=6.5;; C15) H=0.15; LC=1.3;; esac
D=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python3 -B $R/r463/rq463.py sgn=0 L=8 s0=3.0 seed=$2 Tt=300 ell=4 ck=5 ac=0.25 h=$H lc=$LC >> log 2>&1 && touch done
