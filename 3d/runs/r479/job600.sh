#!/bin/bash
# (479Б) bash job600.sh SEED — продовження α0,25 L24 з ck до T600
d=/mnt/project-files/model/s52/r479/L24_a0.25_seed$1; cd $d; [ -f done600 ] && exit 0
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python3 -B /mnt/project-files/model/s52/r351/rq351f.py L=24 s0=3.0 seed=$1 Tt=600 ell=4 ck=5 ac=0.25 >> log 2>&1 && touch done600
