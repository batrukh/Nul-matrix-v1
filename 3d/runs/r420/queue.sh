#!/bin/bash
cd /mnt/project-files/model/s52/r420; seq 1 8 | xargs -P4 -L1 bash job.sh; touch all.done
