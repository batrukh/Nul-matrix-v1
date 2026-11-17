#!/bin/bash
# (349) тест пінінгу: стріли ґратки (без ell), база 8.11 без Φ★, без гальма γ0, s0 3,5, L96. bash job3.sh SEED [T]
R=/mnt/project-files/model/s52/r348; T=${2:-300}; d=lat_ge0_L96_s$1; mkdir -p $R/$d $R/ck/$d; cd $R/$d
[ -f done ] && exit 0
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 $R/code/run268.py nex=0 ge=0 le=8 L=96 n=100 s0=3.5 seed=$1 beta=1.3 Tt=$T snp=10 ck=10 ckdir=$R/ck/$d >> log 2>&1 && touch done
