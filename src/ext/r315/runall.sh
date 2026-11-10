#!/bin/bash
R=/mnt/project-files/model/s52/r315
for j in 0.3 0.45 0.5 0.565; do bash $R/job.sh $j & done; wait
for j in 0 0.7; do bash $R/job.sh $j & done; wait
touch $R/done.flag
