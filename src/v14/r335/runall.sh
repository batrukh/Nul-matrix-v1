#!/bin/bash
cd /mnt/project-files/model/s52/r335
cat jobs.txt | xargs -P 4 -L 1 bash -c 'bash job.sh $0 $1 $2' > runall.log 2>&1
touch all.done
