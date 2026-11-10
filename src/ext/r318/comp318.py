"""(318) А, крок 2: нитки змагаються за бюджет вузла. Граф кандидатів: точки (трикутна ґратка зі зсувом jit або випадкові),
пари на відстані < rc. Вузол v ділить бюджет 1 між своїми нитками: x[v,e] ≥ 0, Σ_e x[v,e] = 1 (місцеве збереження).
Нитка u–v: I = sqrt(x[u,e] x[v,e]) (MODE prod — нитці потрібні обидва кінці) або (x[u,e] + x[v,e])/2 (MODE sum).
Комірка = трикутник кандидатів, амплітуда Π_e sqrt(I_e); максимізуємо F = Σ_комірок амплітуда (= −E) експоненційним
градієнтом на симплексах вузлів. Нитки з I < поріг відмирають — лишається вибране сусідство."""
import sys, json, numpy as np
from scipy.spatial import cKDTree
A = dict(L=20, jit=0.3, rc=1.75, mode="prod", it=3000, eta=0.5, seed=1, out="run")
for a in sys.argv[1:]: k, v = a.split("="); A[k] = type(A[k])(v)
L = A["L"]; n = L*L; rng = np.random.default_rng(A["seed"])
i, j = np.divmod(np.arange(n), L); P = np.c_[(i + j/2) % L, j*np.sqrt(3)/2]; H = L*np.sqrt(3)/2
P = (P + A["jit"]*rng.uniform(-0.5, 0.5, (n, 2))) % [L, H] if A["jit"] < 1 else rng.uniform(0, 1, (n, 2))*[L, H]
T = cKDTree(P, boxsize=[L, H]); E = np.array(sorted(T.query_pairs(A["rc"])))   # ребра-кандидати u<v
m = len(E); eid = {(int(a), int(b)): k for k, (a, b) in enumerate(E)}; nb = [set() for _ in range(n)]
for a, b in E: nb[a].add(b); nb[b].add(a)
tri = np.array([(eid[(a, b)], eid[(b, c) if b < c else (c, b)], eid[(a, c)]) for a, b in E for c in nb[a] & nb[b] if c > b])
deg0 = np.bincount(E.ravel(), minlength=n)
# x[e,0] — частка від E[e,0], x[e,1] — від E[e,1]; старт: порівну
x = np.stack([1/deg0[E[:, 0]], 1/deg0[E[:, 1]]], 1) * (1 + 0.01*rng.standard_normal((m, 2)))
def norm(x):
    s = np.bincount(E[:, 0], x[:, 0], n) + np.bincount(E[:, 1], x[:, 1], n); return x / np.stack([s[E[:, 0]], s[E[:, 1]]], 1)
x = norm(x)
def I_of(x): return np.sqrt(x[:, 0]*x[:, 1]) if A["mode"] == "prod" else x.mean(1)
hist = []
for it in range(A["it"] + 1):
    I = I_of(x) + 1e-300; amp = np.sqrt(I[tri]).prod(1); F = amp.sum()
    dI = np.zeros(m)                                     # ∂F/∂I_e = Σ_комірок amp/(2 I_e)
    for c in range(3): dI += np.bincount(tri[:, c], amp/(2*I[tri[:, c]]), m)
    if A["mode"] == "prod": g = dI[:, None]*I[:, None]/(2*x + 1e-300)
    else: g = np.repeat(dI[:, None]/2, 2, 1)
    if it % 100 == 0:
        alive = I > 1e-3; k = np.bincount(E[alive].ravel(), minlength=n)
        hist.append(dict(it=it, F=F/n, alive=int(alive.sum()), kmean=float(k.mean()), f6=float((k == 6).mean()), kmax=int(k.max())))
        if it % 500 == 0: print(hist[-1], flush=True)
    # експоненційний градієнт на симплексі вузла (масштаб — середній градієнт вузла)
    sc = (np.bincount(E[:, 0], g[:, 0]*x[:, 0], n) + np.bincount(E[:, 1], g[:, 1]*x[:, 1], n))   # Σ x g по вузлу
    gn = g / (np.stack([sc[E[:, 0]], sc[E[:, 1]]], 1) + 1e-300)
    x = norm(x*np.exp(A["eta"]*(gn - 1)))
I = I_of(x); alive = I > 1e-3; k = np.bincount(E[alive].ravel(), minlength=n)
# компоненти живого графа
import scipy.sparse as sp, scipy.sparse.csgraph as cg
G = sp.coo_matrix((np.ones(alive.sum()), (E[alive, 0], E[alive, 1])), (n, n)); nc, lab = cg.connected_components(G, directed=False)
cs = np.bincount(lab); 
res = dict(args=A, hist=hist, kdist=np.bincount(k).tolist(), ncomp=int(nc), comp_sizes=np.bincount(cs).tolist(), F_final=float(F/n),
           cand_deg=float(deg0.mean()), I=I.tolist(), E=E.tolist(), P=P.tolist())
json.dump(res, open(A["out"] + ".json", "w"))
print("final", hist[-1], "components", nc, "size hist", {s: int(c) for s, c in enumerate(np.bincount(cs)) if c}, "kdist", np.bincount(k).tolist(), flush=True)
