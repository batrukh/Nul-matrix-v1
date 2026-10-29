"""(175) порівняння dt = 0,0025 (r175/dt_s*) з dt = 0,005 (r173/A_s*): час першого світу, кількість світів, площа, когерентність, кипіння. Вихід cmp175.txt."""
import json, glob, numpy as np
import os; S = os.environ.get("NM_RUNS", os.path.dirname(os.path.abspath(__file__)) + "/../runs/out/")
def ld(p): return json.load(open(glob.glob(S + p)[0]))["w2"]
for s in (1, 2, 3, 4):
    for dt, p in (("0.005", f"r173/A_s{s}/*_sn_eg0.01.json"), ("0.0025", f"r175/dt_s{s}/*_eg0.01.json")):
        w = ld(p); L = [q for q in w if q["t"] >= 20]; e = w[-1]
        print(s, dt, next((q["t"] for q in w if q["nuc"] > 0), None), e["nuc"], e["nB"], e["big"], round(np.mean([q["hol"] for q in L]), 3), round(np.mean([q["boil"] for q in L]), 3))
