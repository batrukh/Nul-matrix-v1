#!/bin/bash
# bash job.sh DIR GE
R=/mnt/project-files/model/s52/r353/lyap; C=/mnt/project-files/model/s52/r348/code; cd $R/$1; [ -f done ] && exit 0
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 $C/run268.py nex=0 ge=$2 le=8 L=96 n=100 s0=3.5 seed=1 ell=16 iso=1 beta=1.3 Tt=620 snp=1 ck=10 ckdir=$R/$1/ck >> log 2>&1 && touch done
