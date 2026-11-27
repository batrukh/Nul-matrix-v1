#!/bin/bash
R=/mnt/project-files/model/s52/r463
while [ ! -f $R/../r462/all.done ]; do sleep 60; done
xargs -P4 -L1 bash $R/job.sh < $R/jobs.txt
[ $(ls $R/runs/*/done 2>/dev/null | wc -l) -eq 24 ] && touch $R/all.done
