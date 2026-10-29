#!/bin/bash
source "$(dirname "$0")/common.sh"; d=$OUT/light_pf$1_s$2; mkdir -p $d; cd $d
python3 $SRC/run145.py L=96 n=100 s0=3.5 Tt=200 sk=4 ln=0 pf=$1 ck=5 seed=$2 >> log 2>&1
