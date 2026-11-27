#!/bin/bash
cd /mnt/project-files/model/s52/r420; (for s in 1 2 3 4 5 6 7 8; do echo C $s; done; for s in 1 2 3 4 5 6 7 8; do echo F $s; done) | xargs -P4 -L1 bash job2.sh; touch all2.done
