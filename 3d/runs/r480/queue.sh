#!/bin/bash
cd /mnt/project-files/model/s52/r480; xargs -P3 -L1 bash job.sh < ${1:-jobs1.txt}; touch ${1:-jobs1.txt}.done
