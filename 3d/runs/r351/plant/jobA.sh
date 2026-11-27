#!/bin/bash
# (355b) посадка світу в 3D при малому обертанні: ψ-куля R=PR у найбільшу пласку ділянку кінцевого знімка (355) α=AC, T30. bash jobA.sh S0 PR AC
R=/mnt/project-files/model/s52/r351; d=$R/plant/a$3_s$1_pr$2; mkdir -p $d; cd $d; [ -f done ] && exit 0
S=$1; [ "$S" = "3.0" ] && S=3
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python3 $R/rq351.py L=16 s0=$1 seed=1 Tt=30 ell=4 ck=5 fp=0.5 pr=$2 ac=$3 init=$R/alpha/s$1_a$3/t3_s${S}_L16_s1_ell4 >> log 2>&1 && touch done
