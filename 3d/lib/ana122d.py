"""ana122d: середнє за t ≥ 100 (знімки (122)): вихорів усього і поза стінкою вакуумів (жоден вузол у колі r ≤ 1 не має іншого вакууму)."""
import json, glob, re, numpy as np, phi90 as PH
L = 96; X, Y = PH.coords(L)
def node(p): return np.unravel_index(int(np.argmin((X - p[0])**2 + (Y - p[1])**2)), (L, L))
D1 = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
out = {}
for f in sorted(glob.glob("*_sn_gk1_gd_g104.json")):
    m = re.search(r"_s(\d)_.*?(?:_b([\d.]+))?_pf", f); k = f"{m.group(2) or '1.5'}_{m.group(1)}"
    tr = json.load(open(f))["track"]; sn = np.load(f.replace("_g104.json", "_sn.npz")); ts = sn["t"]; NP = sn["npos"] >= 3
    a, b = [], []
    for r in tr:
        if r["t"] < 100: continue
        V = NP[int(np.argmin(abs(ts - r["t"])))]; n_off = 0
        for p in r["cl"]:
            i, j = node(p); vals = {V[(i + x) % L, (j + y) % L] for x, y in D1}; n_off += len(vals) == 1
        a.append(len(r["cl"])); b.append(n_off)
    out[k] = (float(np.mean(a)), float(np.mean(b))); print(k, "усього", round(np.mean(a), 1), "поза стінкою", round(np.mean(b), 1))
json.dump(out, open("ana122d.json", "w"))
