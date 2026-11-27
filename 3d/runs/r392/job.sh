#!/bin/bash
# (392)/(393) bash job.sh KIND SEED ; KIND=late (392) або off (393)
R=/mnt/project-files/model/s52; d=$R/r392/runs/$1_seed$2; mkdir -p $d; cd $d; [ -f done ] && exit 0
if [ $1 = late ]; then SRC=$R/r373/a0_seed$2; AC=0.1; else SRC=$R/r374/a0.1_seed$2; AC=0; fi
START=$SRC/t3_s3_L16_s$2_ell4 START_CK=$R/r392/s0ck/s$2_ck.npz OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python3 -B $R/r388/rq388.py sgn=0 L=16 s0=3.0 seed=$2 Tt=300 ell=4 ck=5 ac=$AC >> log 2>&1 && touch done
