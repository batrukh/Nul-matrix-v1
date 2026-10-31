"""life166: lifetime of a planted vortex (gv=100) in an elastic domain: we follow the nearest cluster (step ≤ 3 per unit) from the planting point."""
import json, glob, numpy as np
def wd(a, b, L=96):
    from math import hypot
    import sys, os; sys.path.insert(0, ".."); import matter98 as M; return float(M.wrapd(np.array([a[0]]), np.array([a[1]]), b[0], b[1], L)[0][0])
out = {}
for d in sorted(glob.glob("p*_s*")):
    J = json.load(open(glob.glob(d + "/*_g104.json")[0])); tr = J["track"]; pos = J["plant"]["xy"]; t_end = 100.0; path = []
    for r in tr:
        if r["t"] <= 100: continue
        if not r["cl"]: break
        dd = [wd(c, pos) for c in r["cl"]]; k = int(np.argmin(dd))
        if dd[k] > 3: break
        pos = r["cl"][k][:2]; t_end = r["t"]; path.append(round(dd[k], 1))
    out[d] = dict(life=round(t_end - 100, 1), depth=round(J["plant"]["depth"], 1), maxstep=max(path) if path else None)
    print(d, out[d])
json.dump(out, open("life174.json", "w"))
