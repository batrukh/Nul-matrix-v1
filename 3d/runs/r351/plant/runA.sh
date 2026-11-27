#!/bin/bash
until [ -f /mnt/project-files/model/s52/r356/night.done ] || ! pgrep -f "r356/night.sh" >/dev/null; do sleep 30; done
cd /mnt/project-files/model/s52/r351/plant; printf "3.0 4 0.3\n3.0 6 0.3\n3.5 4 0.3\n3.5 6 0.3\n" | xargs -P4 -L1 bash jobA.sh >> runA.log 2>&1; [ $(ls a0.3_*/done | wc -l) -ge 4 ] && touch allA.done
