cd /mnt/project-files/model/s52/r395
xargs -P4 -L1 bash job.sh < jobs.txt
n=$(ls runs/*/done 2>/dev/null | wc -l); [ "$n" -ge 8 ] && touch all.done
