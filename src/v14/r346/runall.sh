#!/bin/bash
R=/mnt/project-files/model/s52/r346; cd $R
xargs -P4 -L1 bash $R/job.sh < jobs.txt >> runall.log 2>&1; touch all.done
