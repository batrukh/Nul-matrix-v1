#!/bin/bash
# (480) bash job.sh PR SEED
d=/mnt/project-files/model/s52/r480/R$1_s$2; mkdir -p $d; cd $d; [ -f done ] && exit 0
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python3 -B /mnt/project-files/model/s52/r480/rq480.py L=24 s0=3.0 seed=$2 Tt=40 ell=4 ck=5 ac=0 init=/mnt/project-files/model/s52/r480/snap$2/t3_s3_L24_s$2_ell4 pr=$1 >> log 2>&1 && touch done
