#!/bin/bash
# 16.11 08:35: (385) α0,3 (продовження з ck) → (386)/(387) (автор 08:28 «Так») → решта jobs_v8 (A, B)
cd /mnt/project-files/model/s52/r375; xargs -P4 -I{} sh -c '{}' < jobs_v10.txt && touch all10.done
