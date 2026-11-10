#!/bin/bash
R=/mnt/project-files/model/s52/r321
for s in 1 2; do for d in 0 1; do [ -f $R/s${s}_d$d/done.flag ] || TT=800 bash $R/job.sh $s $d & done; done; wait
