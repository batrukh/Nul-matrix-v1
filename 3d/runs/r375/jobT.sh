#!/bin/bash
# (375) bash jobT.sh AC SEED TT
R=/mnt/project-files/model/s52; d=$R/r375/a$1_seed$2_T$3; mkdir -p $d; cd $d; [ -f done ] && exit 0
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python3 -B $R/r351/rq351f.py L=16 s0=3.0 seed=$2 Tt=$3 ell=4 ck=5 ac=$1 >> log 2>&1 && touch done
