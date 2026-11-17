#!/bin/bash
# (335) база 8.11 (P0,16 ℓ8 γ0,0025, b0,1, β1,3, ℓ_τ16 iso), розгортка запасу і L. bash job.sh S0 L SEED [T]
R=/mnt/project-files/model/s52/r335; T=${4:-300}; d=s$1_L$2_s$3; mkdir -p $R/$d $R/ck/$d; cd $R/$d
[ -f done ] && exit 0
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 /mnt/project-files/model/s52/r268/code/run268.py nex=0 pb=0.1 fine=0.5 ft0=30 ge=0.0025 le=8 L=$2 n=100 s0=$1 seed=$3 ell=16 iso=1 beta=1.3 pf=1 gz=1 gk=1 gd=1 gv=100 Tt=$T ck=10 ckdir=$R/ck/$d >> log 2>&1 && touch done
