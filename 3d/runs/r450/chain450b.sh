#!/bin/bash
# (450) світи на 8 ґратках R; абсолютні шляхи (у chain450.sh відносні ламались, коли cd на мережевий диск не вдавався). Рестарт тим самим.
R=/mnt/project-files/model/s52
seq 1 8 | xargs -P4 -I{} bash $R/r450/job450.sh {}
[ $(ls $R/r450/runs/*/done 2>/dev/null | wc -l) -eq 8 ] && touch $R/r450/all.done
