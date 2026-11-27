#!/bin/bash
R=/mnt/project-files/model/s52/r458
xargs -P4 -L1 bash $R/jobAM.sh < $R/jobsA100.txt
[ $(ls $R/runs/done_A_* 2>/dev/null | wc -l) -eq 24 ] && touch $R/allA100.done
