#!/bin/bash
# чекає кінця (374), потім (375)
until [ -f /mnt/project-files/model/s52/r374/all.done ]; do sleep 30; done
cd /mnt/project-files/model/s52/r375; xargs -P4 -I{} sh -c '{}' < jobs.txt && touch all.done
