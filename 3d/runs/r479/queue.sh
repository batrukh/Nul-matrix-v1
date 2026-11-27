#!/bin/bash
# (479) ідемпотентна черга: повторний запуск лише дораховує (done пропускаються, ck → resume)
cd /mnt/project-files/model/s52/r479; xargs -P4 -I{} sh -c '{}' < jobs.txt; [ $(ls */done 2>/dev/null | wc -l) -ge 24 ] && touch all.done
