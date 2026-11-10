cd "$(dirname "$0")"
python3 braid.py 48 8 3 72 1 x48 > log_x48 2>&1 &
python3 braid.py 48 8 3 144 2 f48 > log_f48 2>&1 &
wait
python3 braid.py 64 10 3 90 1 x64 > log_x64 2>&1 &
python3 braid.py 48 8 1e9 72 1 x48glob > log_x48glob 2>&1 &
wait
echo done > done.txt
