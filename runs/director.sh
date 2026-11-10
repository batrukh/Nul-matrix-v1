#!/bin/bash
# Domain birth with a live director n (Sec. 4.9, Fig. 12): base of Sec. 4.7 (P = 0.16, b = 0.1, beta = 1.3, l_tau = 16), L = 96,
# director coupling mu = 0.03, stiffness k = 0.09. bash runs/director.sh SEED [T] [D]
# T = 800 for seeds 1-2, 300 for seeds 3-6 (paper). D = 0 switches the director off (control). Snapshots of domains and defects every time unit.
source "$(dirname "$0")/common.sh"
S=$1; T=${2:-300}; D=${3:-1}; d=$OUT/director/s${S}_d${D}_T$T; mkdir -p $d/ck; cd $d
if [ "$D" = 1 ]; then export DMU=0.03 DK=0.09; fi
export SNAP=1 DST=20
python3 $SRC/director/run268.py nex=0 pb=0.1 ge=0.0025 le=8 L=96 n=100 s0=3.5 seed=$S ell=16 iso=1 beta=1.3 pf=1 gz=1 gk=1 gd=1 Tt=$T ck=10 ckdir=$d/ck >> log 2>&1
