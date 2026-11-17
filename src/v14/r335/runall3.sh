#!/bin/bash
# (337) нічна пачка статистики: L192 +4, 3,5 T600 +4, вікно 3,0/4,0/4,25 +6, 6,86 +2. Перезапуск тим самим скриптом продовжує з ck.
cd /mnt/project-files/model/s52/r335
cat jobs3.txt | xargs -P 4 -L 1 bash -c 'bash job.sh $0 $1 $2 $3' >> runall3.log 2>&1
touch all3.done
