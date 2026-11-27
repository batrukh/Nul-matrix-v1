#!/bin/bash
cd /mnt/project-files/model/s52/r480; until [ -f jobs1b.txt.done ]; do sleep 30; done; xargs -P4 -L1 bash job2.sh < jobs2.txt; touch jobs2.txt.done
