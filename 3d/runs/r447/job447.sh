#!/bin/bash
[ "$1" = world ] && [ -f /mnt/project-files/model/s52/r447/HOLD ] && exit 0
# bash job447.sh lat PROT SEED  |  bash job447.sh world PROT SEED
R=/mnt/project-files/model/s52/r447; cd $R; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
if [ "$1" = lat ]; then python3 -B pfc447.py $2 $3 >> runs/lat_$2_s$3.log 2>&1; exit; fi
d=$R/runs/w_$2_s$3; mkdir -p $d; cd $d; [ -f done ] && exit 0
python3 -B $R/rq447.py lat=$R/runs/lat_$2_s$3.npz seed=$3 Tt=300 s0=3.0 ac=0.1 ck=5 early=0.7 >> log 2>&1 && touch done
