#!/bin/bash
xargs -P4 -L1 bash /mnt/project-files/model/s52/r458/job.sh < /mnt/project-files/model/s52/r458/jobs.txt
[ $(ls /mnt/project-files/model/s52/r458/runs/done_* 2>/dev/null | wc -l) -eq 24 ] && touch /mnt/project-files/model/s52/r458/all.done
