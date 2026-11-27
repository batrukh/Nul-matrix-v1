cd /mnt/project-files/model/s52/r425 && xargs -P4 -L1 bash job.sh < jobs.txt >> queue.log 2>&1; touch all.done
