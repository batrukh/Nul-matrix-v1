#!/bin/bash
# (179) run.sh TAG HT PT — переаналіз кадрів (173) A_s1..4 з іншими порогами
cd /mnt/project-files/model/s52/r179
PYTHONPATH=/mnt/project-files/model/s52/r173/code NUMBA_NUM_THREADS=1 HT=$2 PT=$3 OUTF=sel_$1.json SNPAT='/mnt/project-files/model/s52/r173/A_s{sd}/lc_s03.5_n100_L96_s{sd}_sk4l0_sn_sn.npz' python3 sel179.py > sel_$1.log 2>&1
