#!/bin/bash
# (335) доповнення: край вікна 4,25 і природний запас 6,86; стартує, коли основна пачка закінчиться
cd /mnt/project-files/model/s52/r335
while [ ! -f all.done ]; do sleep 60; done
cat jobs2.txt | xargs -P 4 -L 1 bash -c 'bash job.sh $0 $1 $2' > runall2.log 2>&1
touch all2.done
