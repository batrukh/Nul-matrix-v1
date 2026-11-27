"""kib103: речовина з Матриці — механізм Кіббла. Знімки local60 pf (раз на 5 одиниць часу): ψ, Φ★, нитки z.
(а) Z₂-дефекти протополя (трикутники з det голономії площини < 0) лише всередині світу (ψ > 0,3 у всіх трьох вершинах), кластери, у часі.
(б) Де народжується дефект: стан Матриці на тому місці за 5 і 10 одиниць до появи (|sin h| ниток, інтенсивність |z|², ψ)
    проти бездефектних трикутників того ж світу в ту саму мить.
(в) Дві реалізації шуму Φ★ на тій самій Матриці (pseed 0/1): чи збігаються місця дефектів (частка збігів у межах 4 проти випадкового розміщення)."""
import numpy as np, json, sys
import matter98 as M, phi90 as PH
from simNlib import Fabric

def load(name):
    z = np.load(name + "_snap.npz"); return z["t"], z["psi"], z["P"], z["z"]

def tri_world(psi, L, thr=0.3):
    p = psi.reshape(L, L); sh = lambda d: np.roll(p, (-d[0], -d[1]), axis=(0, 1))
    up = (p > thr) & (sh((1, 0)) > thr) & (sh((0, 1)) > thr); dn = (sh((1, 0)) > thr) & (sh((1, 1)) > thr) & (sh((0, 1)) > thr)
    return np.array([up, dn])

def analyse(name, L=96):
    t, psi, P, z = load(name); F = Fabric(L); X, Y = PH.coords(L); out = []
    hs = [np.abs(np.sin(F.holonomy(zz.astype(complex)))).reshape(L, L, 2).transpose(2, 0, 1) for zz in z]
    inten = []
    for zz in z:
        a = np.abs(zz)**2; nd = np.zeros(L*L); c = np.zeros(L*L)
        np.add.at(nd, F.tail, a); np.add.at(nd, F.head, a); np.add.at(c, F.tail, 1); np.add.at(c, F.head, 1); inten.append((nd/c).reshape(L, L))
    for k in range(len(t)):
        w = tri_world(psi[k], L); det, _ = M.holonomy(P[k].astype(float)); d = (det < 0) & w
        cl = M.clusters(np.where(d, -1.0, 1.0), X, Y, L)
        out.append(dict(t=float(t[k]), wtri=int(w.sum()), ndef=int(d.sum()), ncl=len(cl), cl=cl, mask=d, wnode=psi[k].reshape(L, L) > 0.3))
    # (б) народження: кластер, якого не було в попередньому знімку в межах 4
    births = []
    for k in range(1, len(out)):
        for c in out[k]["cl"]:
            if all(np.hypot(*(np.subtract(c, q))) > 4 for q in out[k - 1]["cl"]):
                births.append((k, c))
    rows = []
    for k, c in births:
        rr = M.wrapd(X, Y, c[0], c[1], L)[0]; near = rr < 2.5
        w = tri_world(psi[k], L); far = w.any(0) & (rr > 6) & ~out[k]["mask"].any(0)
        rec = dict(t=out[k]["t"], c=c)
        for lag in (1, 2):
            if k - lag < 0: continue
            hh = hs[k - lag].mean(0); ii = inten[k - lag]; pp = psi[k - lag].reshape(L, L)
            rec[f"h{lag}"] = (float(hh[near].mean()), float(hh[far].mean())); rec[f"I{lag}"] = (float(ii[near].mean()), float(ii[far].mean()))
            rec[f"psi{lag}"] = (float(pp[near].mean()), float(pp[far].mean()))
        rows.append(rec)
    return out, rows

def match(oa, ob, L=96, R=4.0, seed=0):
    """частка дефектів реалізації a, що мають дефект реалізації b ближче R, проти випадкових точок у світі b (30 перемішувань)"""
    X, Y = PH.coords(L); rng = np.random.default_rng(seed); res = []
    near = lambda p, B: min(float(M.wrapd(np.array(q[0]), np.array(q[1]), p[0], p[1], L)[0]) for q in B) < R
    for a, b in zip(oa, ob):
        if not a["cl"] or not b["cl"]: continue
        f0 = np.mean([near(p, b["cl"]) for p in a["cl"]])
        wn = np.flatnonzero(b["wnode"].ravel()); sh = []
        for _ in range(30):
            pk = rng.choice(wn, len(b["cl"])); B = list(zip(X.ravel()[pk], Y.ravel()[pk])); sh.append(np.mean([near(p, B) for p in a["cl"]]))
        res.append(dict(t=a["t"], na=len(a["cl"]), nb=len(b["cl"]), f=float(f0), frand=float(np.mean(sh))))
    return res

if __name__ == "__main__":
    names = sys.argv[1:]; allr = {}
    for nm in names:
        out, rows = analyse(nm); allr[nm] = (out, rows)
        print(nm); print(" t/ncl/wtri:", [(o["t"], o["ncl"], o["wtri"]) for o in out if o["wtri"]])
        for r in rows: print("  birth", r)
    if len(names) == 2:
        for m in match(allr[names[0]][0], allr[names[1]][0]): print(" match", m)
    json.dump({nm: dict(series=[{k: v for k, v in o.items() if k not in ("mask", "wnode")} for o in out], births=rows) for nm, (out, rows) in allr.items()},
              open("kib103_" + str(len(names)) + ".json", "w"))
