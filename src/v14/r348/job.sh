#!/bin/bash
# (348) контроль без поля Φ★ (pf=0) для статті про домени: згрубіння з гальмом і без. bash job.sh GE L SEED [T]; база 8.11 без Φ★, s0 3,5, знімки snp=10
R=/mnt/project-files/model/s52/r348; T=${4:-300}; d=ge$1_L$2_s$3; mkdir -p $R/$d $R/ck/$d; cd $R/$d
[ -f done ] && exit 0
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 $R/code/run268.py nex=0 ge=$1 le=8 L=$2 n=100 s0=3.5 seed=$3 ell=16 iso=1 beta=1.3 Tt=$T snp=10 ck=10 ckdir=$R/ck/$d >> log 2>&1 && touch done
