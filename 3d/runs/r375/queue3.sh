#!/bin/bash
# (375) з (377) після C — порядок: C T600, (377) оживлення, A, B
cd /mnt/project-files/model/s52/r375; xargs -P4 -I{} sh -c '{}' < jobs_v3.txt && touch all.done
