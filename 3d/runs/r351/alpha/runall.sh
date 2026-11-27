#!/bin/bash
# чекає кінця прогонів доменів (r353), потім 8 прогонів α
until [ -f /mnt/project-files/model/s52/r353/all.done ] || ! pgrep -f "r353/job.sh" >/dev/null; do sleep 30; done
cd /mnt/project-files/model/s52/r351/alpha; xargs -P4 -L1 bash job.sh < jobs.txt >> runall.log 2>&1; [ $(ls */done | wc -l) -ge 8 ] && touch all.done
