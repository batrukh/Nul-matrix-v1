#!/bin/bash
# (447) після нічної черги: ґратки PFC (16), далі світи (16). Безпечно перезапускати.
R=/mnt/project-files/model/s52; cd $R/r447; mkdir -p runs
while [ ! -f $R/night1120.done ]; do sleep 60; done
for p in S Q; do for s in 1 2 3 4 5 6 7 8; do echo "lat $p $s"; done; done | xargs -P4 -L1 bash job447.sh
for p in S Q; do for s in 1 2 3 4 5 6 7 8; do echo "world $p $s"; done; done | xargs -P4 -L1 bash job447.sh
ls runs/w_*_s*/done | wc -l | grep -qx 16 && touch all447.done
