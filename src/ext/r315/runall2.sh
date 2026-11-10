#!/bin/bash
R=/mnt/project-files/model/s52/r315
while pgrep -f "jit=0.3 " >/dev/null || pgrep -f "jit=0.45 seed=1 Tt" >/dev/null || pgrep -f "jit=0.5 " >/dev/null || pgrep -f "jit=0.565 " >/dev/null; do sleep 20; done
bash $R/job.sh 0 1 0 & bash $R/job.sh 0.4 1 0 & bash $R/job.sh 0.45 1 1 & bash $R/job.sh 0.3 2 0 & wait
touch $R/done2.flag
