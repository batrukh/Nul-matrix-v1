#!/bin/bash
# 16.11 14:20: після (388) Б/В → (389) народження → решта (A/B 375 у кінці; автор 14:14 «Згоден»)
R=/mnt/project-files/model/s52
until [ -f $R/r388/C/C1/done ] && [ -f $R/r388/C/C0/done ] && [ $(ls $R/r388/B/*/done 2>/dev/null | wc -l) -ge 10 ]; do sleep 20; done
pkill -f queue12.sh; pkill -f "xargs -P4"; sleep 1; pkill -f "r351/rq351f.py"; sleep 2
cd $R/r375; xargs -P4 -I{} sh -c '{}' < jobs_v13.txt && touch all13.done
