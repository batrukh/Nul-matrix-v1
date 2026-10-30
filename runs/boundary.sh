#!/bin/bash
# (181) photon transmission through a straight domain boundary: Matrix weight FLOOR, wavenumber K, boundary width DELTA.
# Usage: boundary.sh FLOOR K DELTA   (paper: FLOOR 0.02/0.005, K 0.3/0.6/1.2, DELTA 1/3)
source "$(dirname "$0")/common.sh"; d=$OUT/boundary; mkdir -p $d; cd $d
PYTHONPATH=$SRC python3 $SRC/imp181.py $1 $2 $3 >> log 2>&1
