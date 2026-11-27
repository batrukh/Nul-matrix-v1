"""(449) острови і поле стріл τ (21.11): чи острів сидить у особливому місці τ. Готові знімки r425 (α0,25, T600, Кун L16)."""
import sys, glob, json, numpy as np
sys.path.insert(0, "/mnt/project-files/model/s52"); import fabgen as G
L = 16; n = L**3
idx = np.arange(n).reshape(L, L, L)
nbr = [np.roll(idx, s, a).ravel() for a in range(3) for s in (1, -1)]
def comp(B):
    lab = -np.ones(n, int); cs = []
    for v in np.where(B)[0]:
        if lab[v] >= 0: continue
        st = [v]; lab[v] = len(cs); c = [v]
        while st:
            u = st.pop()
            for w in (x[u] for x in nbr):
                if B[w] and lab[w] < 0: lab[w] = len(cs); st.append(w); c.append(w)
        cs.append(np.array(c))
    return sorted(cs, key=len, reverse=True)
def feats(tau):
    T = tau.reshape(L, L, L); g = np.sqrt(sum(((np.roll(T, -1, a) - np.roll(T, 1, a))/2)**2 for a in range(3))).ravel()
    lap = (sum(np.roll(T, 1, a) + np.roll(T, -1, a) for a in range(3)) - 6*T).ravel()
    return dict(tau=tau, grad=g, lap=lap)
rng = np.random.default_rng(0); res = []
for d in sorted(glob.glob("/mnt/project-files/model/s52/r425/runs/*_seed*")):
    tag = d.split("/")[-1]; seed = int(tag.split("seed")[1]); h = {"M": 0.36, "S": 0.39}.get(tag.split("_")[0], None) or float(tag.split("_")[0][1:])
    fs = glob.glob(d + "/t3_*.npz")
    if not fs: continue
    psi = np.load(fs[0])["psi"]; B = psi > 0; frac = B.mean()
    if not (50 <= B.sum() <= 0.4*n): res.append(dict(tag=tag, seed=seed, h=h, kind="world" if frac > 0.4 else "none", frac=float(frac))); continue
    cs = comp(B); isl = cs[0]; m = np.zeros(n, bool); m[isl] = True
    F = feats(G.tau_cubic(L, 4.0, seed)); out = dict(tag=tag, seed=seed, h=h, kind="island", size=int(len(isl)))
    M = m.reshape(L, L, L)
    for k, v in F.items():
        obs = v[m].mean(); V = v.reshape(L, L, L)
        null = np.array([V[np.roll(M, tuple(rng.integers(0, L, 3)), (0, 1, 2))].mean() for _ in range(2000)])
        out[k] = round(float(obs), 3); out[k + "_pct"] = round(float((null < obs).mean()), 3)
    out["cells"] = m
    res.append(out)
# стабільність місця острова між h для того самого зерна
isl = [r for r in res if r["kind"] == "island"]
for a in isl:
    a["overlap"] = {b["tag"]: round(float((a["cells"] & b["cells"]).sum()/min(a["cells"].sum(), b["cells"].sum())), 2) for b in isl if b["seed"] == a["seed"] and b is not a}
for r in res: r.pop("cells", None); print(r)
json.dump(res, open("ana449.json", "w"), indent=1)
