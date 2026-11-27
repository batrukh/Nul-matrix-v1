#!/bin/bash
# C → (377) → (378) полотно → (379) Φ★ → (381) скан s0 α0,2 → A → B (рішення автора 15.11 16:15 Київ)
cd /mnt/project-files/model/s52/r375; xargs -P4 -I{} sh -c '{}' < jobs_v6.txt && touch all.done
