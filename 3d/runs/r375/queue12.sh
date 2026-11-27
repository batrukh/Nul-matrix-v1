#!/bin/bash
# 16.11 13:20: чекає кінця (388А), тоді (388Б) першими (автор 12:46: А → Б → В), далі решта jobs_v10
A=/mnt/project-files/model/s52/r388/A
until [ $(ls $A/*/done 2>/dev/null | wc -l) -ge 6 ]; do sleep 20; done
pkill -f queue11.sh; pkill -f "xargs -P4"; sleep 1; pkill -f "r351/rq351f.py"; sleep 2
cd /mnt/project-files/model/s52/r375; xargs -P4 -I{} sh -c '{}' < jobs_v12.txt && touch all12.done
