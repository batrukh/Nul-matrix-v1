#!/bin/bash
source "$(dirname "$0")/common.sh"
if [ -n "$3" ]; then d=$OUT/matter_p$1_s$2; X="gv=$3"; else d=$OUT/matter_b$1_s$2; X=""; fi
mkdir -p $d/ck; cd $d
python3 $SRC/run166.py ge=0.01 le=8 L=96 n=100 s0=3.5 seed=$2 sk=4 ln=0 beta=$1 pf=1 gz=1 gk=1 gd=1 $X Tt=250 ck=10 ckdir=$d/ck >> log 2>&1
