#!/bin/bash
# budget scan WITHOUT elasticity (grey points of Fig. 2), e.g.: window.sh sk=4 ln=0 seed=1 s0=3.0
source "$(dirname "$0")/common.sh"; tag=$(echo "$@" | tr ' =' '_'); d=$OUT/window_$tag; mkdir -p $d; cd $d
python3 $SRC/local60.py L=96 n=100 Tt=250 snp=10 "$@" > log 2>&1
