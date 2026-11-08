#!/bin/bash
# Variant of Sec. 4.7 (P = 0.16: l = 8, gamma = 0.0025; b = 0.1; beta = 1.3; random-field arrows l_tau = 16), L = 96, T = 600,
# a vortex planted at t = 100 (gv=100). Output and checkpoints go where src/realw286.py expects them. bash runs/variant.sh SEED
source "$(dirname "$0")/common.sh"
V=$OUT/variant; d=$V/g0.0025_s$1_T600; mkdir -p $d $V/ck/g0.0025_s$1_T600; cd $d
python3 $SRC/variant/run268.py nex=0 pb=0.1 fine=0.5 ft0=30 ge=0.0025 le=8 L=96 n=100 s0=3.5 seed=$1 ell=16 iso=1 beta=1.3 pf=1 gz=1 gk=1 gd=1 gv=100 Tt=600 ck=10 ckdir=$V/ck/g0.0025_s$1_T600 >> log 2>&1
