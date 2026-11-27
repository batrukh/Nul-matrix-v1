#!/bin/bash
cd /mnt/project-files/model/s52/r374; xargs -P4 -I{} sh -c '{}' < jobs.txt && touch all.done
