#!/bin/bash
cd /mnt/project-files/model/s52/r448; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for m in R Z N; do for s in 1 2 3 4 5 6 7 8; do echo "$m $s"; done; done | xargs -P4 -L1 sh -c 'python3 -B pfc448.py $0 $1 0.25 2000 >> runs/$0_s$1.log 2>&1'
ls runs/*_e0.25_s*.json | wc -l | grep -qx 24 && touch all.done
