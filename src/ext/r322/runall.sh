#!/bin/bash
R=/mnt/project-files/model/s52/r322
for s in 3 4 5 6; do [ -f $R/s$s/done.flag ] || bash $R/job.sh $s & done; wait
