#!/bin/bash
source "$(dirname "$0")/common.sh"; d=$OUT/birth_s$1; mkdir -p $d; cd $d
python3 $SRC/run145.py L=96 n=100 s0=3.5 Tt=${2:-100} sk=4 ln=0 sn=1 ck=5 seed=$1 >> log 2>&1
