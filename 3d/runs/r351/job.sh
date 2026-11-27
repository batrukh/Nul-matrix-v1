#!/bin/bash
# (351) 3D крок 0: bash job.sh S0 SEED [T]; тетраедри Куна L16, база 8.11 (β1,3, ℓ_τ 4, без Φ★ і гальма); resume з ck
R=/mnt/project-files/model/s52/r351; T=${3:-100}; d=$R/runs/s$1_seed$2; mkdir -p $d; cd $d
[ -f done ] && exit 0
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python3 $R/rq351.py L=16 s0=$1 seed=$2 Tt=$T ell=4 ck=5 >> log 2>&1 && touch done
