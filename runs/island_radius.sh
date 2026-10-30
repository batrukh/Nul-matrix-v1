#!/bin/bash
# (184) as island.sh, but only random link phases, in disks of radius 9 and 12 (run island.sh first: it writes the event list)
source "$(dirname "$0")/common.sh"; d=$OUT/island; cd $d
export R183_SN=$OUT/birth_s{sd}/lc_s03.5_n100_L96_s{sd}_sk4l0_sn_sn.npz PYTHONPATH=$SRC
python3 $SRC/mk184.py 9 && python3 $SRC/mk184.py 12
for s in rs/*_scrR*.npz; do b=$(basename $s .npz); [ -f psi/$b.npz ] && continue
  sd=${b%%_*}; t=$(python3 -c "import numpy as np;print(float(np.load('$s')['t'])+15)"); mkdir -p w/$b; (cd w/$b
  python3 $SRC/run145.py L=96 n=100 s0=3.5 Tt=$t sk=4 ln=0 sn=1 seed=$sd rs=$d/$s > log 2>&1
  python3 -c "import numpy as np,glob;a=np.load(glob.glob('*_sn_sn.npz')[0]);np.savez_compressed('$d/psi/$b.npz',t=a['t'],psi=a['psi'].astype(np.float16))") && rm -rf w/$b
done
