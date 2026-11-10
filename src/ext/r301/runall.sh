#!/bin/bash
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1
python3 probe.py comp 1e-3 400 > l_comp 2>&1 &
python3 probe.py comp 0 400 > l_comp_bg 2>&1 &
python3 probe.py alice 1e-3 400 > l_alice 2>&1 &
python3 probe.py alice 0 400 > l_alice_bg 2>&1 &
wait
python3 probe.py vac 1e-3 400 > l_vac 2>&1
