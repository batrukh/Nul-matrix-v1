#!/bin/bash
R=/mnt/project-files/model/s52/r458
while [ ! -f $R/all.done ] && pgrep -f "rq458.py [0-9.]+ [0-9]+ 200" >/dev/null && ! pgrep -f ALIVE >/dev/null; do sleep 30; done
xargs -P4 -L1 bash $R/jobAM.sh < $R/jobsAM.txt
[ $(ls $R/runs/done_A_* 2>/dev/null | wc -l) -eq 16 ] && touch $R/allA.done
