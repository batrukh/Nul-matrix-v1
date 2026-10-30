#!/bin/bash
# (182) photon leak over long times: the packet test of light.sh states, run to T (default 300). Usage: leak.sh STATE_g91.npz FLOOR [T]
source "$(dirname "$0")/common.sh"; d=$OUT/leak; mkdir -p $d; cd $d
PH_T=${3:-300} python3 $SRC/ph182.py $1 $2 >> log 2>&1
