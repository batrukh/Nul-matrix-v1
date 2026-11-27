cd /mnt/project-files/model/s52/r470g
for a in "2.9 16 0.2" "3.0 16 0.2" "3.15 16 0.2" "3.0 16 -0.2" "2.9 24 0.2"; do set -- $a
 for sd in 1 2 3 4 5 6 7 8; do [ -f lin_s$(python3 -c "print('%g'%$1)")_L$2_mu$3_s$sd.json ] || python3 -u lin470g.py $1 $2 $sd $3 > log_$1_$2_$3_$sd.txt 2>&1; done; done
echo done > all.done
