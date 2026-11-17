#!/bin/bash
# (346) згрубіння з пружним гальмом і без. bash job.sh GE L SEED [T]; база 8.11, s0 3,5, знімки snp=10
R=/mnt/project-files/model/s52/r346; T=${4:-600}; d=ge$1_L$2_s$3; mkdir -p $R/$d $R/ck/$d; cd $R/$d
[ -f done ] && exit 0
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 $R/code/run268.py nex=0 pb=0.1 fine=0.5 ft0=30 ge=$1 le=8 L=$2 n=100 s0=3.5 seed=$3 ell=16 iso=1 beta=1.3 pf=1 gz=1 gk=1 gd=1 gv=100 Tt=$T snp=10 ck=10 ckdir=$R/ck/$d >> log 2>&1 && touch done
