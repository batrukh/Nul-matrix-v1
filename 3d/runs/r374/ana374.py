"""ana374: (а) стійкість світу α0,1/0,2 до T300 (L16); (б) L24 α0,1 T100: частка ψ>0, перколяція найбільшого кластера ψ<0, кількість зародків."""
import sys; sys.path.insert(0, "/mnt/project-files/model/s52")
import numpy as np, json, os, glob
import fabgen as G
R = "/mnt/project-files/model/s52/r374"; out = {}

def wrap_cluster(f, L, B):
    """кластери B з перевіркою, чи кластер обгортає тор (перколяція) — BFS з розгорнутими координатами"""
    nb = [[] for _ in range(f.n)]
    for a, b in zip(f.tail, f.head): nb[a].append(b); nb[b].append(a)
    ijk = f.ijk.astype(int); seen = np.zeros(f.n, bool); pos = np.zeros((f.n, 3), int); res = []
    for v in np.where(B)[0]:
        if seen[v]: continue
        st = [v]; seen[v] = True; pos[v] = ijk[v]; c = 0; wr = set()
        while st:
            u = st.pop(); c += 1
            for w in nb[u]:
                if not B[w]: continue
                d = (ijk[w] - ijk[u] + L//2) % L - L//2; p = pos[u] + d
                if not seen[w]: seen[w] = True; pos[w] = p; st.append(w)
                else:
                    dw = p - pos[w]
                    for ax in range(3):
                        if dw[ax] != 0: wr.add(ax)
        res.append((c, sorted(wr)))
    return sorted(res, key=lambda x: -x[0])

# (а)
f16 = [None]
for a in ["0.1", "0.2"]:
    rows = []
    for s in range(1, 9):
        fn = f"{R}/a{a}_seed{s}/t3_s3_L16_s{s}_ell4.json"
        if not os.path.exists(fn): continue
        w = json.load(open(fn))["w2"]; bt = {round(r["t"]): r["big"] for r in w if "big" in r}
        b100, b300 = bt.get(100, 0), bt.get(300, 0); bmax = max(bt.values())
        ps = np.load(fn[:-5] + ".npz", allow_pickle=True)["psi"]
        if f16[0] is None: f16[0] = G.FabricG(16, "tet")
        ng = wrap_cluster(f16[0], 16, ps <= 0); pp = wrap_cluster(f16[0], 16, ps > 0)
        rows.append(dict(seed=s, neg_big=ng[0][0] if ng else 0, neg_wrap=ng[0][1] if ng else [], neg_ncl=len(ng), pos_wrap=pp[0][1] if pp else [], b100=b100, b200=bt.get(200, 0), b300=b300, bmax=bmax, f300=b300/4096,
                         shrink=(1 - b300/b100) if b100 else None, world300=b300 >= 50,
                         q300=w[-1].get("qhol"), nuc300=w[-1].get("nuc"),
                         big_t=[bt.get(t, 0) for t in range(100, 301, 10)]))
    out["a" + a] = rows
    n = len(rows); nw = sum(r["world300"] for r in rows); st = sum(r["world300"] and r["shrink"] is not None and r["shrink"] <= 0.2 for r in rows)
    print(f"α{a}: n={n} world@300 {nw}/{n}, stable(≤20% shrink) {st}/{n}")
    for r in rows: print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "big_t"})

# (б)
rowsL = []; f = None
for s in range(1, 9):
    fn = f"{R}/L24_seed{s}/t3_s3_L24_s{s}_ell4.npz"
    if not os.path.exists(fn): continue
    if f is None: f = G.FabricG(24, "tet")
    psi = np.load(fn, allow_pickle=True)["psi"]; w = json.load(open(fn[:-4] + ".json"))["w2"]
    pos = wrap_cluster(f, 24, psi > 0); neg = wrap_cluster(f, 24, psi <= 0)
    tb = next((r["t"] for r in w if r.get("big", 0) >= 50), None)
    rowsL.append(dict(seed=s, fpos=float((psi > 0).mean()), npos_cl=len([c for c in pos if c[0] >= 5]),
                      big_pos=pos[0][0] if pos else 0, big_pos_wrap=pos[0][1] if pos else [],
                      big_neg=neg[0][0] if neg else 0, big_neg_wrap=neg[0][1] if neg else [], nneg_cl=len(neg),
                      tbirth=tb, nuc_max=max(r.get("nuc", 0) for r in w), q_end=w[-1].get("qhol"),
                      big_t=[r.get("big", 0) for r in w]))
    r = rowsL[-1]; print("L24", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "big_t"})
out["L24"] = rowsL
json.dump(out, open(f"{R}/ana374.json", "w"), indent=1)
