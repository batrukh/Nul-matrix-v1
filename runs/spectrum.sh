#!/bin/bash
# (180) gauge spectrum in the Matrix: plane waves in a uniform Phi = 0 Matrix (weights 1 and 0.02), then the local mass of each mode on saved v1 states.
# Usage: spectrum.sh [SNAPSHOT_g91.npz ...]   (default: data/r173/B_pf1_s1; other states from light.sh)
source "$(dirname "$0")/common.sh"; d=$OUT/spectrum; mkdir -p $d; cd $d
export PYTHONPATH=$SRC
python3 $SRC/mat180.py 24 > mat180.log 2>&1
for f in "${@:-$H/../data/r173/B_pf1_s1/lc_s03.5_n100_L96_s1_sk4l0_pf1_g91.npz}"; do python3 $SRC/mass180.py "$f" > /dev/null; done
