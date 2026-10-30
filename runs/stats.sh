#!/bin/bash
# (178) statistics in model v1 (Sec. 4.3, Fig. 4 "Statistics"): stats.sh TAG L S0 SEED GAMMA TT
#   window:   stats.sh win_s3.5_seed4 72 3.5 4 0.01 150      (s0 2.7-3.5, seeds 4-8)
#   arrest:   stats.sh P0.29_seed1 72 3.5 1 0.0045312 150     (gamma = P/64, ell = 8)
#   merging:  stats.sh ost_g0_seed1 96 3.5 1 0 300
source "$(dirname "$0")/common.sh"
d=$OUT/stats/$1; mkdir -p $d; cd $d
python3 $SRC/run145.py L=$2 n=100 s0=$3 Tt=$6 snp=25 sk=4 ln=0 seed=$4 eg=$5 el=8 ck=5 >> log 2>&1
echo done >> log
