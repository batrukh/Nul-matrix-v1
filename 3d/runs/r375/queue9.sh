#!/bin/bash
# 16.11 рішення автора 06:21: «кольори» (385) першими, потім решта jobs_v8 (A, B); чекає завершення 4 прогонів α0,05
for p in 21011 21024 21055 21073; do while kill -0 $p 2>/dev/null; do sleep 30; done; done
cd /mnt/project-files/model/s52/r375; xargs -P4 -I{} sh -c '{}' < jobs_v9.txt && touch all9.done
