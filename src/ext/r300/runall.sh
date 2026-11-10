#!/bin/bash
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1
run(){ python3 aw.py $1 $2 200 > l_$1_$2 2>&1; }
run both 0 & run both 4 & run both 8 & run both 16 & wait
run alice 0 & run wind 4 & run wind 16 & wait
