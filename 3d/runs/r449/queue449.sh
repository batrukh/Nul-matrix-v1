#!/bin/bash
cd /mnt/project-files/model/s52/r449
xargs -P4 -L1 bash job449.sh < jobs449.txt
[ $(ls runs/*/done 2>/dev/null | wc -l) -eq 32 ] && touch all.done
