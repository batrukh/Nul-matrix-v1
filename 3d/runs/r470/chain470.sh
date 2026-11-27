#!/bin/bash
cd /mnt/project-files/model/s52/r470
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cat jobs.txt | xargs -P 4 -L 1 bash -c 'python3 /mnt/project-files/model/s52/r470/pm470.py $0 $1 $2 $3 $4 $5 >> /mnt/project-files/model/s52/r470/runs/log_$0_$4_$1.txt 2>&1'
touch /mnt/project-files/model/s52/r470/all.done
