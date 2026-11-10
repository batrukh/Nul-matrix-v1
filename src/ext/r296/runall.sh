#!/bin/bash
cd /mnt/project-files/model/s52/r296
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 T=30
printf "vort 2.5\nvort 3.5\nvort 3\nctrl 3\nctrl 2\nctrl 4\nctrl 3.5\n" | xargs -P4 -L1 bash -c 'python3 spin.py $0 $1 > log_$0_$1 2>&1'
echo ALLDONE > done.txt
