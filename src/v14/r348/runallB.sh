#!/bin/bash
# (348b): продовження (348) до T600 + вікно запасу без поля. Перезапуск тим самим скриптом продовжує з ck.
R=/mnt/project-files/model/s52/r348; cd $R
xargs -P4 -L1 sh -c 'bash $0/$1 $2 $3 $4 $5' $R < jobsB.txt >> runallB.log 2>&1; touch allB.done
