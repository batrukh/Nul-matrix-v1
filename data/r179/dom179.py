"""(179) чутливість детектора світу: поріг ψ (−0,2/0/+0,2) і мінімальний розмір кластера (3/5/10/20) на кінцевих станах вікна (176)+(178), L72."""
import numpy as np, glob, json
from scipy.ndimage import label
L = 72; STR = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]])
def sizes(M):
    lb, n = label(np.tile(M, (3, 3)), structure=STR); c = lb[L:2*L, L:2*L]; ids, cnt = np.unique(c[c > 0], return_counts=True); return cnt
R = "/mnt/project-files/model/s52"; out = {}
for s0 in ("2.7", "3.0", "3.3", "3.5"):
    fs = glob.glob(f"{R}/r178/win_s{s0}_seed*/lc_*_sk4l0.npz") + glob.glob(f"{R}/r176/s{s0}_seed*/lc_*_sk4l0.npz")
    for pt in (-0.2, 0.0, 0.2):
        for mn in (3, 5, 10, 20):
            N, F, MED = [], [], []
            for f in fs:
                ps = np.load(f)["psi"].reshape(L, L); c = sizes(ps > pt); c = c[c >= mn]
                N.append(len(c)); F.append(100*c.sum()/L**2); MED.append(float(np.median(c)) if len(c) else 0)
            out[f"{s0}|{pt}|{mn}"] = dict(n=[min(N), max(N)], frac=[round(min(F), 1), round(max(F), 1)], med=round(float(np.median(MED)), 1), runs=len(fs))
            print(s0, pt, mn, out[f"{s0}|{pt}|{mn}"])
json.dump(out, open("dom179.json", "w"), indent=0)
