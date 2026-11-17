#!/bin/bash
R=/mnt/project-files/model/s52/r348; cd $R
xargs -P4 -L1 bash $R/job3.sh < jobsC.txt >> runallC.log 2>&1; touch allC.done
