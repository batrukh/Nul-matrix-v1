cd /mnt/project-files/model/s52/r448; mkdir -p smoke; cd smoke; mkdir -p runs
export OMP_NUM_THREADS=1
for e in 0.25 0.26; do python3 -B ../pfc448.py Z 1 $e 1000 > z$e.log 2>&1 & done
python3 -B ../pfc448.py N 1 0.26 1000 > n.log 2>&1 &
wait
