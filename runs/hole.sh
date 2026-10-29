#!/bin/bash
# (177) hole in the substrate (Sec. 4.6, Fig. 7): hole.sh L S0 SEED R Q   e.g. hole.sh 48 4.5 1 8 0.6   (R=0: control)
source "$(dirname "$0")/common.sh"
d=$OUT/hole_L$1/s$2_seed$3_R$4_q$5; mkdir -p $d/ck; cd $d
python3 $SRC/run145.py L=$1 n=100 s0=$2 Tt=60 snp=5 sk=4 ln=0 seed=$3 ck=10 pR=$4 pq=$5 >> log 2>&1
echo done >> log
