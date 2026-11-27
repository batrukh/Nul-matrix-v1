#!/bin/bash
# (420Б) bash job2.sh KIND SEED ; KIND=F (латка R8 у киплячому хаосі, REG) або C (контроль без хаосу); продовження з ck
R=/mnt/project-files/model/s52; d=$R/r420/runs/${1}_L24_seed$2; [ -f $d/done ] && exit 0; mkdir -p $d; cd $d
if [ "$1" = F ]; then E="CHAOS=F R=8 COMP=0 REG=1"; else E="CHAOS=F R=100 COMP=0 REG=0"; fi
env $E EARLY=0.8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python3 -B $R/r420/rq420.py sgn=0 L=24 s0=3.0 seed=$2 Tt=300 ell=4 ck=5 ac=0.1 >> log 2>&1 && touch done
