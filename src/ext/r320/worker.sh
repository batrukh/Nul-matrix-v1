#!/bin/bash
# (320) робітник: для кожного k зі списку — релаксація, потім гессіан (пропускає готове). bash worker.sh R NS k1 k2 ...
cd /mnt/project-files/model/s52/r320; R=$1; NS=$2; shift 2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for k in "$@"; do
  T=L48_R${R}_A0.5_w2_N${NS}_k$k
  [ -f fin/$T.npz ] || python3 relax.py 48 $R $NS $k > logs/relax_$T.log 2>&1
  [ -f modes/$T.npz ] || python3 hess.py $T 12 0.03 > logs/hess_$T.log 2>&1
  echo "$T готово $(date +%H:%M)" >> logs/progress.log
done
