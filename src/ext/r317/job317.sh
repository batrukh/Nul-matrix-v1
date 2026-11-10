#!/bin/bash
# bash job317.sh MODE LAM [L=50] [SWEEPS=400] [T0=0.003]
cd /mnt/project-files/model/s52/r317; o=${1}_lam${2}_L${3:-50}
python3 graph317.py mode=$1 lam=$2 L=${3:-50} sweeps=${4:-400} T0=${5:-0.003} out=$o > $o.log 2>&1
