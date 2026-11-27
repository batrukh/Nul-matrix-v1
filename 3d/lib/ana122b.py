"""ana122b: що передує народженню Z₂-вихору (знімки щоодиниці, (122)). Для кожного народження (вакуум-свідомий детектор) і для
випадкових вузлів світу в ту саму мить: (а) окіл r ≤ 2 був поза світом (ψ < 0,3) хоч в одну з 3 попередніх одиниць — «щойно приєднаний»;
(б) у околі r ≤ 3 є межа вакуумів Φ★/−Φ★ зараз або одиницю тому; (в) глибина місця; (г) окіл r ≤ 3 перетинає «злиття»: вузли, що
одиницю тому були в різних світах (компонентах), тепер в одному."""
import json, glob, re, sys, numpy as np, scipy.ndimage as ndi, matter98 as M, phi90 as PH
L = 96; X, Y = PH.coords(L)
def dd(p, q): return float(M.wrapd(np.array(p[0]), np.array(p[1]), q[0], q[1], L)[0])
def node(p): return np.unravel_index(int(np.argmin((X - p[0])**2 + (Y - p[1])**2)), (L, L))
def disk(r):
    a = np.arange(-r, r + 1); I, J = np.meshgrid(a, a, indexing="ij"); return [(i, j) for i, j in zip(I.ravel(), J.ravel()) if abs(i) + abs(j) + abs(i + j) <= 2*r]   # гексагональний окіл
D2, D3 = disk(2), disk(3)
def around(A, ij, D): i, j = ij; return np.array([A[(i + a) % L, (j + b) % L] for a, b in D])
def labels(W):
    lab, n = ndi.label(np.tile(W, (3, 3))); lab = lab[L:2*L, L:2*L]; return lab
res = {}
for f in sorted(glob.glob("*_sn_gk1_gd_g104.json")):
    m = re.search(r"_s(\d)_.*?(?:_b([\d.]+))?_pf", f); key = f"{m.group(2) or '1.5'}_{m.group(1)}"
    tr = json.load(open(f))["track"]; sn = np.load(f.replace("_g104.json", "_sn.npz")); ts = sn["t"]; PS = sn["psi"].reshape(len(ts), L, L); NP = sn["npos"]
    rng = np.random.default_rng(0); B = {k: [] for k in ("new", "wall", "merge", "dep")}; R = {k: [] for k in B}
    for a, c in zip(tr, tr[1:]):
        if c["t"] < 30: continue
        k = int(np.argmin(abs(ts - c["t"])))
        if k < 3: continue
        W = PS[k] > 0.3; wall = np.zeros((L, L), bool); wallp = np.zeros((L, L), bool)
        for d in ((1, 0), (0, 1), (-1, 1)):
            for V, out in ((NP[k], wall), (NP[k - 1], wallp)):
                s_ = np.roll(V, (-d[0], -d[1]), (0, 1)); out |= (V != s_)
        Wb = W & np.roll(W, (-1, 0), (0, 1)) & np.roll(W, (0, -1), (0, 1))
        wall &= W; wallp &= PS[k - 1] > 0.3
        out3 = np.zeros((L, L), bool)
        for m_ in (1, 2, 3): out3 |= PS[k - m_] < 0.3
        lab0 = labels(PS[k - 1] > 0.3); lab1 = labels(W)
        dep = ndi.distance_transform_edt(np.tile(PS[k] > 0, (3, 3)))[L:2*L, L:2*L]
        def feats(ij):
            l0 = around(lab0, ij, D3); l1 = around(lab1, ij, D3); mrg = len(set(l0[l0 > 0])) > 1 and len(set(l1[l1 > 0])) == 1
            return dict(new=bool(around(out3, ij, D2).any()), wall=bool(around(wall | wallp, ij, D3).any()), merge=bool(mrg), dep=float(dep[ij]))
        births = [p for p in c["cl"] if all(dd(p, q) > 2.5 for q in a["cl"])]
        for p in births:
            for kk, v in feats(node(p)).items(): B[kk].append(v)
        idx = np.argwhere(W)
        for ij in idx[rng.choice(len(idx), min(len(idx), 5), replace=False)] if len(idx) else []:
            for kk, v in feats(tuple(ij)).items(): R[kk].append(v)
    res[key] = dict(nbirth=len(B["new"]), **{f"b_{k}": float(np.mean(v)) if v else None for k, v in B.items()}, **{f"r_{k}": float(np.mean(v)) for k, v in R.items()},
                    b_dep_med=float(np.median(B["dep"])) if B["dep"] else None, r_dep_med=float(np.median(R["dep"])))
    print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[key].items()}, flush=True)
json.dump(res, open("ana122b.json", "w"))
