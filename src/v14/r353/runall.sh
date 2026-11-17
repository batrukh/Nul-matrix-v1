#!/bin/bash
R=/mnt/project-files/model/s52/r353; cd $R
xargs -P4 -L1 bash $R/job.sh < jobs.txt >> runall.log 2>&1; [ $(ls */done | wc -l) -ge 20 ] && touch all.done
