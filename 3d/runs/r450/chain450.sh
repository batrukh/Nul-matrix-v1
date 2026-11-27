#!/bin/bash
# чекає r449/all.done, далі світи на 8 ґратках R; рестарт тим самим (ck, done)
cd /mnt/project-files/model/s52
while [ ! -f r449/all.done ]; do sleep 60; done
seq 1 8 | xargs -P4 -I{} bash r450/job450.sh {}
[ $(ls r450/runs/*/done 2>/dev/null | wc -l) -eq 8 ] && touch r450/all.done
