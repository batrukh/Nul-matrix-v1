#!/bin/bash
# (348b) вікно запасу без поля Φ★: bash job2.sh S0 SEED [T]; база 8.11 без Φ★, γ0,0025, L96, знімки snp=10
R=/mnt/project-files/model/s52/r348; T=${3:-300}; d=w$1_L96_s$2; mkdir -p $R/$d $R/ck/$d; cd $R/$d
[ -f done ] && exit 0
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 $R/code/run268.py nex=0 ge=0.0025 le=8 L=96 n=100 s0=$1 seed=$2 ell=16 iso=1 beta=1.3 Tt=$T snp=10 ck=10 ckdir=$R/ck/$d >> log 2>&1 && touch done
