#!/bin/bash
# (452Г) bash runall.sh — пропускає готові; перезапуск = знову bash runall.sh
cd /mnt/project-files/model/s52/r452/M16
cat jobs.txt | xargs -P4 -L1 bash -c 'f=lock_a$(python3 -c "print(f\"{float($0):g}\")")_s$1_L16.json; [ -f $f ] || OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python3 -B ../lock452m.py $0 $1 16 >> log 2>&1'
ls lock_*_L16.json | wc -l > all.count; touch all.done
