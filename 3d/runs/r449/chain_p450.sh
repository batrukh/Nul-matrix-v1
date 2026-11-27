#!/bin/bash
# дороблює контроль ×10 (z+ψ, runs_a) → виправлений контроль (ψ×10, runs_p) → світи (450). Рестарт тим самим.
cd /mnt/project-files/model/s52
seq 1 8 | xargs -P4 -I{} bash r449/job449a.sh 5 {} 10
seq 1 8 | xargs -P4 -I{} bash r449/job449p.sh 5 {} 10
[ $(ls r449/runs_p/*/done 2>/dev/null | wc -l) -eq 8 ] && touch r449/p.done
[ -f r449/p.done ] && bash r450/chain450.sh
