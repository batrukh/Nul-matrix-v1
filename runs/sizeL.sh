#!/bin/bash
# model v1 at system size L: sizeL.sh L SEED  (Fig. 4d: L = 48, 72, 96, 144)
source "$(dirname "$0")/common.sh"; d=$OUT/sizeL_L$1_s$2; mkdir -p $d; cd $d
python3 $SRC/ela139.py g=0.01 ell=8 L=$1 n=100 s0=3.5 Tt=250 snp=25 sk=4 ln=0 seed=$2 > log 2>&1
