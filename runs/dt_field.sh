#!/bin/bash
# (187) time-step check with the field sector: as light.sh but dt = 0.0025 (field step halves too). Usage: dt_field.sh PF SEED; compare with light.sh PF SEED
source "$(dirname "$0")/common.sh"; d=$OUT/dtfield_pf$1_s$2; mkdir -p $d; cd $d
python3 $SRC/run145.py L=96 n=100 s0=3.5 Tt=200 sk=4 ln=0 pf=$1 ck=5 dt=0.0025 seed=$2 >> log 2>&1
