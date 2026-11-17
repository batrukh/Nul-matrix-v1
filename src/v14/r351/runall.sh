#!/bin/bash
cd /mnt/project-files/model/s52/r351; xargs -P4 -L1 bash job.sh < jobs.txt; n=$(ls runs/*/done 2>/dev/null | wc -l); echo "done $n/12"; [ $n -eq 12 ] && touch all.done
