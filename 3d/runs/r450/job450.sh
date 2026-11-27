#!/bin/bash
R=/mnt/project-files/model/s52; d=$R/r450/runs/s$1; mkdir -p $d; cd $d; [ -f done ] && exit 0
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python3 -B $R/r447/rq447c.py lat=$R/r448/runs/R_e0.25_s$1.npz seed=$1 Tt=300 ac=0.1 ck=5 early=0.7 >> log 2>&1 && touch done
