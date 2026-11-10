#!/bin/bash
# (321): bash job.sh SEED D  — база P0,16 (як (283), без латок), T 250; D=1 з директором (DMU 0,03, DK 0,09), D=0 без. Resume тим самим.
R=/mnt/project-files/model/s52/r321; d=s$1_d$2; mkdir -p $R/$d $R/ck/$d; cd $R/$d
if [ "$2" = 1 ]; then export DMU=0.03 DK=0.09; fi
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 $R/code/run268.py nex=0 pb=0.1 ge=0.0025 le=8 L=96 n=100 s0=3.5 seed=$1 ell=16 iso=1 beta=1.3 pf=1 gz=1 gk=1 gd=1 Tt=${TT:-250} ck=10 ckdir=$R/ck/$d >> log 2>&1
echo done > done.flag
