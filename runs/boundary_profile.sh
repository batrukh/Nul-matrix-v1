#!/bin/bash
# (188) profile across a domain boundary from the snapshots of light.sh (run light.sh 1 1, 1 2, 0.3 1, 0.3 2 first). Usage: boundary_profile.sh
source "$(dirname "$0")/common.sh"; cd $OUT && R188_SNAP=$OUT/light_pf{pf}_s{sd}/lc_s03.5_n100_L96_s{sd}_sk4l0_pf{pf}_snap.npz python3 $SRC/prof188.py
