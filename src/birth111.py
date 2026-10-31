"""(111): search for the nucleation "trigger" in 2D. Frames local60 sn=1 (every time unit). A birth is a new domain (≥ 5 nodes with ψ > 0) that does not overlap
the domains of the previous moment. For birth sites and for control Matrix nodes (farther than 6 from all domains and from the birth site) at the same moment
t_b − k (k = 2, 5, 10) we compare local node features. The measure is AUC (0.5 — the feature says nothing; ≈ 1 or ≈ 0 — a strong trigger)."""
import os; os.environ["NB"] = "0"
import numpy as np, json, sys, local60 as LC, birth48 as BR
from scipy.ndimage import distance_transform_edt as edt

def fab_at(s, f0t, f0h, tail):
    f = s.fab; fl = tail != f0t; f.tail = tail; f.head = np.where(fl, f0t, f0h)
    for k in ("1", "2", "3"):
        E = getattr(f, "E" + k); setattr(f, "S" + k, np.where(fl[E], -s._S0[k], s._S0[k]))
    return f

def features(s, f, z, psi, zp, L):
    n = f.n; a_, b_ = f.tail, f.head; A2 = np.abs(z)**2; cnt = np.bincount(a_, None, n) + np.bincount(b_, None, n)
    nodeavg = lambda v: (np.bincount(a_, v, n) + np.bincount(b_, v, n))/cnt
    hol = f.holonomy(z); TV = f.TV; tn = np.zeros(n); tc = np.zeros(n); h2 = np.zeros(n)
    for j in range(3): np.add.at(tn, TV[:, j], np.abs(hol)); np.add.at(tc, TV[:, j], 1); np.add.at(h2, TV[:, j], hol**2)
    holm = tn/tc; holsd = np.sqrt(np.maximum(h2/tc - (tn/tc)**2, 0))
    od = np.bincount(a_, None, n)                                         # links leaving the node (along the arrow)
    P = s.F(z, np.zeros(n), parts=True); cost = P["cost"]
    dA = A2 - np.abs(zp)**2                                               # change of intensity per unit time (boiling)
    boil = nodeavg(np.abs(dA)); sync = np.abs(nodeavg(dA))/np.maximum(boil, 1e-9)   # 1 — all links of the node change in step
    ph = np.angle(z*np.conj(zp)); phsync = np.hypot(nodeavg(np.cos(ph)), nodeavg(np.sin(ph)))   # coherence of phase rotations
    Avar = nodeavg(A2**2) - nodeavg(A2)**2
    return dict(psi=psi, A2=nodeavg(A2), A2sd=np.sqrt(np.maximum(Avar, 0)), hol=holm, holsd=holsd, extr=np.abs(od - 3.0), cost=cost,
                boil=boil, sync=sync, phsync=phsync)

def auc(x, y):
    """AUC: probability that a random birth node has a larger feature value than a random control node"""
    r = np.argsort(np.argsort(np.r_[x, y])) + 1; return float((r[:len(x)].sum() - len(x)*(len(x) + 1)/2)/(len(x)*len(y)))

def analyse(nm, L=96, lags=(1, 2, 5, 10)):
    d = np.load(nm + "_sn.npz"); T = d["t"]; PS = d["psi"]; Z = d["z"].astype(complex); TL_ = d["tail"]
    s = LC.Loc(L=L, h=0.3, lc=1.3); f = s.fab; f0t, f0h = f.tail.copy(), f.head.copy(); s._S0 = {k: getattr(f, "S" + k).copy() for k in ("1", "2", "3")}
    s.m0 = float(np.mean(np.abs(Z[0])**2))
    nb = [[] for _ in range(f.n)]
    for p, q in zip(f0t, f0h): nb[p].append(q); nb[q].append(p)
    births = []
    for k in range(1, len(T)):
        B = PS[k] > 0; Bp = PS[k - 1] > 0
        seen = np.zeros(f.n, bool)
        for v in np.flatnonzero(B):
            if seen[v]: continue
            comp = [v]; seen[v] = True; st = [v]
            while st:
                u = st.pop()
                for w in nb[u]:
                    if B[w] and not seen[w]: seen[w] = True; comp.append(w); st.append(w)
            if len(comp) >= 5 and not Bp[comp].any(): births.append((k, comp))
    rows = {lag: {"birth": [], "ctrl": []} for lag in lags}
    rng = np.random.default_rng(0)
    for k, comp in births:
        for lag in lags:
            j = k - lag
            if j < 1: continue
            f = fab_at(s, f0t, f0h, TL_[j]); F = features(s, f, Z[j], PS[j], Z[j - 1], L)
            W = (PS[j] > 0).reshape(L, L) | np.isin(np.arange(f.n), comp).reshape(L, L)
            far = edt(~np.tile(W, (3, 3)))[L:2*L, L:2*L].reshape(-1) > 6
            ctrl = rng.choice(np.flatnonzero(far), min(300, far.sum()), replace=False) if far.any() else []
            rows[lag]["birth"] += [{q: float(v[c]) for q, v in F.items()} for c in comp[:30]]
            rows[lag]["ctrl"] += [{q: float(v[c]) for q, v in F.items()} for c in ctrl[:30]]
    return births, rows

if __name__ == "__main__":
    allrows = {}; nb_ = {}
    for nm in sys.argv[1:]:
        births, rows = analyse(nm); nb_[nm] = [(int(k), len(c)) for k, c in births]; print(nm, "births", nb_[nm], flush=True)
        for lag, r in rows.items():
            allrows.setdefault(lag, {"birth": [], "ctrl": []}); allrows[lag]["birth"] += r["birth"]; allrows[lag]["ctrl"] += r["ctrl"]
    res = {}
    for lag, r in allrows.items():
        if not r["birth"]: continue
        keys = r["birth"][0].keys(); res[lag] = {}
        for q in keys:
            x = np.array([b[q] for b in r["birth"]]); y = np.array([c[q] for c in r["ctrl"]])
            res[lag][q] = dict(auc=round(auc(x, y), 3), birth=round(float(x.mean()), 3), ctrl=round(float(y.mean()), 3), nb=len(x))
        print("lag", lag, {q: v["auc"] for q, v in res[lag].items()})
    json.dump(dict(births=nb_, res={str(k): v for k, v in res.items()}), open("birth111.json", "w"))
