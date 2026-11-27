#!/bin/bash
# (479Б) ідемпотентна черга, 4 ядра; перезапуск лише дораховує
cd /mnt/project-files/model/s52/r479; seq 1 8 | xargs -P4 -I{} bash job600.sh {}; [ $(ls L24_a0.25_seed*/done600 2>/dev/null | wc -l) -ge 8 ] && touch all600.done
