#!/bin/bash
cd /mnt/project-files/model/s52/r373
while [ ! -f ../r372/all.done ]; do sleep 60; done
xargs -P4 -L1 bash job.sh < jobs.txt && touch all.done
