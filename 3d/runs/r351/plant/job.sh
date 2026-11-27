#!/bin/bash
# (354) посадка світу в 3D на базі 8.11: ψ-куля радіуса PR у найбільшій пласкій ділянці кінцевого знімка (351), T30, баланс сил fp=0.5. bash job.sh S0 PR
R=/mnt/project-files/model/s52/r351; d=$R/plant/s$1_pr$2; mkdir -p $d; cd $d; [ -f done ] && exit 0
S=$1; [ "$S" = "3.0" ] && S=3; [ "$S" = "4.0" ] && S=4
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python3 $R/rq351.py L=16 s0=$1 seed=1 Tt=30 ell=4 ck=5 fp=0.5 pr=$2 init=$R/runs/s$1_seed1/t3_s${S}_L16_s1_ell4 >> log 2>&1 && touch done
