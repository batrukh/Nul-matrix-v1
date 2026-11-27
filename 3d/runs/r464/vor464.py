"""(464) Правило «нитка = спільна грань комірок Вороного» на ґратках, вирощених з поля (r448 R, Z) — опис ступенів вузлів і тотожності Ейлера."""
import sys, json, numpy as np
sys.path[:0] = ["/mnt/project-files/model/s52/r447", "/mnt/project-files/model/s52"]
import fabpts as F
out = {}
for arm in ("Z", "R"):
    for k in range(1, 9):
        d = np.load(f"/mnt/project-files/model/s52/r448/runs/{arm}_e0.25_s{k}.npz"); P = d["P"]; L = float(np.atleast_1d(d["box"])[0])
        f = F.fromPoints(P, L); deg = np.asarray(f.deg); n = f.n; T = len(f.tets)
        h = np.bincount(deg, minlength=20)
        out[f"{arm}{k}"] = dict(n=n, zmean=float(deg.mean()), TN=T/n, euler=float(2*(1 + T/n)), f14=float((deg == 14).mean()),
                                zmin=int(deg.min()), zmax=int(deg.max()), hist={int(z): int(c) for z, c in enumerate(h) if c})
        r = out[f"{arm}{k}"]; print(arm, k, n, "z̄ %.3f  2(1+T/N) %.3f  z=14 %.3f  [%d..%d]" % (r["zmean"], r["euler"], r["f14"], r["zmin"], r["zmax"]))
json.dump(out, open("vor464.json", "w"), indent=1)
