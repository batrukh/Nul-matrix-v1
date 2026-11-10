#!/bin/bash
# (315) аморфна Матриця: bash job.sh JIT [SEED] — L96 T300, ck кожні 10 од. у r315/ck
R=/mnt/project-files/model/s52/r315; C=/mnt/project-files/model/s52/r268/code; d=j$1_s${2:-1}_g${3:-0}; mkdir -p $R/$d $R/ck/$d; cd $R/$d

NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 $R/run315.py jit=$1 seed=${2:-1} gseed=${3:-0} Tt=300 ck=10 ckdir=$R/ck/$d >> log 2>&1
