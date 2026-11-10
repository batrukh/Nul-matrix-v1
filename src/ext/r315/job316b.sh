#!/bin/bash
# (316б): bash job316b.sh GR GIN [JIT=0.565] [T=150]; GR 0 = без зерна (контроль jit=GIN на всій сітці)
R=/mnt/project-files/model/s52/r315; J=${3:-0.565}; T=${4:-150}; d=gr$1_gin$2_j$J; mkdir -p $R/$d $R/ck/$d; cd $R/$d
if [ "$1" = "0" ]; then A="jit=$2"; else A="jit=$J gr=$1 gin=$2"; fi
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 $R/run316.py $A Tt=$T ck=10 ckdir=$R/ck/$d >> log 2>&1
