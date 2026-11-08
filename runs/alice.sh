#!/bin/bash
# Alice string tests of Sec. 4.7, Fig. 10.
#   bash runs/alice.sh synthetic        uniform synthetic domain L = 128, lambda = 6, 8, 12, vortex and control (285); then src/ana285.py
#   bash runs/alice.sh real SEED        real domain of runs/variant.sh SEED (needs its checkpoint), lambda = 6, 8 (286); then src/ana286.py
source "$(dirname "$0")/common.sh"
if [ "$1" = synthetic ]; then
  d=$OUT/alice_synthetic; mkdir -p $d; cd $d
  for lam in 6 8 12; do for m in vort ctrl; do CPH=0.65 AMP=0.02 SX=$lam SY=12 X0=-28 TR=60 python3 $SRC/abw285.py $m $lam 128 >> log 2>&1; done; done
  python3 $SRC/ana285.py >> log 2>&1
else
  d=$OUT/alice_real; mkdir -p $d; cd $d
  for lam in 6 8; do NM_RUNS=$OUT/variant TR=40 T=45 python3 $SRC/realw286.py $2 $lam >> log 2>&1; done
  python3 $SRC/ana286.py >> log 2>&1
fi
