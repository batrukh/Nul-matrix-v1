#!/bin/bash
# чекає r449/all.done → контроль шуму ×10 (з5, NS1–8) → світи (450). Рестарт тим самим.
cd /mnt/project-files/model/s52
while [ ! -f r449/all.done ]; do sleep 60; done
seq 1 8 | xargs -P4 -I{} bash r449/job449a.sh 5 {} 10
[ $(ls r449/runs_a/*/done 2>/dev/null | wc -l) -eq 8 ] && touch r449/a.done
[ -f r449/a.done ] && bash r450/chain450.sh
