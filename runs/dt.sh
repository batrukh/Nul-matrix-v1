#!/bin/bash
source "$(dirname "$0")/common.sh"; d=$OUT/dt_s$1; mkdir -p $d; cd $d
python3 $SRC/run145.py L=96 n=100 s0=3.5 Tt=100 sk=4 ln=0 ck=5 dt=0.0025 seed=$1 >> log 2>&1
