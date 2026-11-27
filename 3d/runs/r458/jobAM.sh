#!/bin/bash
R=/mnt/project-files/model/s52/r458; cd $R; [ -f runs/done_A_r$1_s$2 ] && exit 0
ALIVE_M=A EOFF=0.185 H=0.36 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python3 -B $R/rq458.py $1 $2 200 >> runs/log_A_r$1_s$2 2>&1 && touch runs/done_A_r$1_s$2
