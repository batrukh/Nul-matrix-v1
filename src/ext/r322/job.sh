#!/bin/bash
# (322): bash job.sh SEED — з директором, T 300, знімки світу (ψ>0,3 бітами) і позиції дефектів щоодиниці (DST 20 кроків поля = 1 од.)
R=/mnt/project-files/model/s52/r322; d=s$1; mkdir -p $R/$d $R/ck/$d; cd $R/$d
export DMU=0.03 DK=0.09 SNAP=1 DST=20
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 $R/code/run268.py nex=0 pb=0.1 ge=0.0025 le=8 L=96 n=100 s0=3.5 seed=$1 ell=16 iso=1 beta=1.3 pf=1 gz=1 gk=1 gd=1 Tt=${TT:-300} ck=10 ckdir=$R/ck/$d >> log 2>&1
echo done > done.flag
