#!/bin/bash
# (353) рецензія статті про домени (v1.1): прохід α (ac), ℓ_τ, нові реалізації. База 8.11 без Φ★, s0 3,5, L96, T300.
# bash job.sh TAG GE SEED ELL AC   (ELL=0 — стріли ґратки)
R=/mnt/project-files/model/s52/r353; C=/mnt/project-files/model/s52/r348/code; d=$1; mkdir -p $R/$d $R/ck/$d; cd $R/$d
[ -f done ] && exit 0
E=""; [ "$4" != "0" ] && E="ell=$4 iso=1"
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 $C/run268.py nex=0 ge=$2 le=8 L=96 n=100 s0=3.5 seed=$3 $E ac=$5 beta=1.3 Tt=300 snp=10 ck=10 ckdir=$R/ck/$d >> log 2>&1 && touch done
