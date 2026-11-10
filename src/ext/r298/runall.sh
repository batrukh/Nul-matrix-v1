#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
(python3 vort.py pair 24 200 > l_pair24 2>&1; echo done > d_pair24) &
(python3 vort.py ctrl 24 200 > l_ctrl24 2>&1; echo done > d_ctrl24) &
(python3 vort.py pair 6 200 > l_pair6 2>&1; echo done > d_pair6) &
wait
