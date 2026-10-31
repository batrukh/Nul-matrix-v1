"""ana166: matter with elasticity versus (120) without elasticity. For each run: vortices (vacuum-aware detector) at t ≥ 150, births
(cluster with no predecessor closer than 2 in the previous frame) per 1000 domain triangles per unit time, domain area, fraction of −Φ★ (f3), number of domains."""
import json, glob, re, numpy as np, sys, os
R120 = os.environ.get("NM_R120", "")   # data of (120) without elasticity; empty — the comparison is skipped
def wd(a, b, L=96):
    d = np.abs(a - b); d = np.minimum(d, L - d); return np.hypot(*d.T) if d.ndim > 1 else np.hypot(*d)
def stats(track, w2=None, t0=150):
    T = [r for r in track if r["t"] >= t0]
    ncl = np.mean([r["ncl"] for r in T]); wt = np.mean([r["wtri"] for r in T]); f3 = np.mean([r["f3"] or 0 for r in T])
    nb = 0; prev = None; tb = 0
    for r in track:
        cur = np.array([c[:2] for c in r["cl"]]) if r["cl"] else np.zeros((0, 2))
        if prev is not None and r["t"] >= t0:
            for c in cur:
                if len(prev) == 0 or wd(prev, c[None, :]).min() > 2: nb += 1
            tb += 1
        prev = cur
    out = dict(ncl=round(ncl, 1), birth=round(1000*nb/max(tb, 1)/max(wt, 1), 2), wtri=int(wt), f3=round(f3, 2), nlate=[r["ncl"] for r in track if r["t"] in (150.0, 200.0, 250.0)])
    if w2: Wl = [r for r in w2 if r["t"] >= t0]; out.update(nuc=round(np.mean([r["nuc"] for r in Wl]), 1), nB=int(np.mean([r["nB"] for r in Wl])), big=int(np.mean([r["big"] for r in Wl])))
    return out
def w2log(fn):
    return [eval(l) for l in open(fn) if l.startswith("{")]
res = {}
for b in ("1.3", "1.5"):
    for s in (1, 2):
        bt = f"_b{b}" if b != "1.5" else ""
        f0 = R120 + f"lc_s03.5_n100_L96_s{s}_tau8_D1_g0.02_iso{bt}_pf1_gz0.02_gk1_gd_g104.json"
        if R120: res[f"g0_b{b}_s{s}"] = stats(json.load(open(f0))["track"], w2log(R120 + f"rungd_b{b}_s{s}.log"))
        g = glob.glob(f"b{b}_s{s}/*_g104.json"); w = glob.glob(f"b{b}_s{s}/*_ge0.01_le8.json")
        if g: res[f"ela_b{b}_s{s}"] = stats(json.load(open(g[0]))["track"], json.load(open(w[0]))["w2"] if w else None)
for k, v in res.items(): print(k, v)
json.dump(res, open("ana174.json", "w"), indent=1)
