#!/bin/bash
# (381) bash job.sh S0 AC SEED
R=/mnt/project-files/model/s52; d=$R/r381/s$1_a$2_seed$3; mkdir -p $d; cd $d; [ -f done ] && exit 0
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python3 -B $R/r351/rq351f.py L=16 s0=$1 seed=$3 Tt=300 ell=4 ck=5 ac=$2 >> log 2>&1 && touch done
