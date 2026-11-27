#!/bin/bash
# 16.11 12:55: (388А) першими (автор 12:46), далі решта jobs_v10 (A/B продовжуються з ck)
cd /mnt/project-files/model/s52/r375; xargs -P4 -I{} sh -c '{}' < jobs_v11.txt && touch all11.done
