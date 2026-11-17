#!/bin/bash
# (353) продовження ge0 (без гальма) з T600 до T1200 — перевірка застигання (рецензія). bash job1200.sh SEED; копія T600 у теці T600/
R=/mnt/project-files/model/s52/r348; d=ge0_L96_s$1; cd $R/$d; [ -f done1200 ] && exit 0
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 $R/code/run268.py nex=0 ge=0 le=8 L=96 n=100 s0=3.5 seed=$1 ell=16 iso=1 beta=1.3 Tt=1200 snp=10 ck=10 ckdir=$R/ck/$d >> log1200 2>&1 && touch done1200
