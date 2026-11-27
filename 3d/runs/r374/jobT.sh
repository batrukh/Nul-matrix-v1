#!/bin/bash
# (374а) bash jobT.sh AC SEED — продовження (373) до T300
R=/mnt/project-files/model/s52; d=$R/r374/a$1_seed$2; cd $d; [ -f done ] && exit 0
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python3 -B $R/r351/rq351f.py L=16 s0=3.0 seed=$2 Tt=300 ell=4 ck=5 ac=$1 >> log 2>&1 && touch done
