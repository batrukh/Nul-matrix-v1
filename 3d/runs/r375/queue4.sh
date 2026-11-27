#!/bin/bash
# (375)+(379): C T600 → (377) оживлення → (378) полотно → (379) Φ★ → A → B (рішення автора 15.11 13:57: Φ★ після полотна)
cd /mnt/project-files/model/s52/r375; xargs -P4 -I{} sh -c '{}' < jobs_v4.txt && touch all.done
