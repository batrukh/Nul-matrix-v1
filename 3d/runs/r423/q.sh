cd /mnt/project-files/model/s52/r423 && xargs -P4 -L1 python3 fdt423.py < jobs.txt > q.log 2>&1; touch all.done
