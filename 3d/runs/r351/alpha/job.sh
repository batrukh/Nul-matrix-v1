#!/bin/bash
# (355) прохід обертання α у 3D на базі 8.11 (згода автора 12.11 «Прохід обертання»). bash job.sh S0 AC
R=/mnt/project-files/model/s52/r351; d=$R/alpha/s$1_a$2; mkdir -p $d; cd $d; [ -f done ] && exit 0
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python3 $R/rq351.py L=16 s0=$1 seed=1 Tt=100 ell=4 ck=5 ac=$2 >> log 2>&1 && touch done
