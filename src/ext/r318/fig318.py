# fig213: (318) нитки змагаються за бюджет вузла
import json, ast, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import scipy.sparse as sp, scipy.sparse.csgraph as cg
C = [("prod_j0.0_rc1.75", "правильна, rc 1,75"), ("prod_j0.3_rc1.5", "зсув 0,3, rc 1,5"), ("prod_j0.3_rc1.75", "зсув 0,3, rc 1,75"), ("prod_j0.3_rc2.2", "зсув 0,3, rc 2,2"),
     ("sum_j0.3_rc1.75", "зсув 0,3, rc 1,75, I = сума"), ("prod_j1.0_rc1.75", "випадкові точки, rc 1,75")]
def last(n): return ast.literal_eval(open(f"{n}.log").read().strip().split("\n")[-1])
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), dpi=150, gridspec_kw=dict(width_ratios=[1.2, 1.1, 1]))
a = ax[0]; y = np.arange(len(C))
a.barh(y + 0.2, [100*last(n)["f6_delaunay"] for n, _ in C], 0.38, color="#9ec5f4", label="Делоне тих самих точок")
a.barh(y - 0.2, [100*last(n)["f6_top"] for n, _ in C], 0.38, color="#0d366b", label="3N найсильніших ниток")
a.set_yticks(y, [l for _, l in C], fontsize=8); a.invert_yaxis(); a.set_xlabel("вузлів із 6 сусідами, %"); a.legend(fontsize=7.5, frameon=False, loc="lower right")
a.set_title("(а) порядок після змагання ниток, N 400", fontsize=10, loc="left")
D = json.load(open("prod_j0.3_rc1.5.json")); I = np.array(D["I"]); E = np.array(D["E"]); P = np.array(D["P"]); n = len(P); L = 20; H = L*np.sqrt(3)/2
al = I > 1e-3; G = sp.coo_matrix((np.ones(al.sum()), (E[al, 0], E[al, 1])), (n, n)); nc, lab = cg.connected_components(G, directed=False); cs = np.bincount(lab)
a = ax[1]
for (u, v), w in zip(E[al], I[al]):
    d = P[v] - P[u]
    if abs(d[0]) > L/2 or abs(d[1]) > H/2: continue
    small = cs[lab[u]] < 20; a.plot(*zip(P[u], P[v]), color="#eb6834" if small else "#1c5cab", lw=0.4 + 3*w/I.max(), alpha=.9)
dead = ~al
for u, v in E[dead]:
    d = P[v] - P[u]
    if abs(d[0]) < L/2 and abs(d[1]) < H/2: a.plot(*zip(P[u], P[v]), color="#ccc", lw=0.4, ls=":")
a.scatter(P[:, 0], P[:, 1], s=4, c="#333", zorder=3); a.set_aspect(1); a.set_xticks([]); a.set_yticks([])
a.set_title("(б) зсув 0,3, rc 1,5: живі нитки (товщина = I);\nпомаранчеве — відірвані тетраедри, сіре — відмерлі", fontsize=9, loc="left")
D = json.load(open("prod_j0.3_rc1.75.json")); I = np.array(D["I"]); E = np.array(D["E"]); P = np.array(D["P"])
ln = np.linalg.norm(((P[E[:, 0]] - P[E[:, 1]] + [L/2, H/2]) % [L, H]) - [L/2, H/2], axis=1)
a = ax[2]; a.scatter(ln, I, s=4, c="#3987e5", alpha=.6, linewidths=0); a.set_yscale("log"); a.set_ylim(1e-6, 1)
a.set_xlabel("довжина нитки (у кроках ґратки)"); a.set_ylabel("інтенсивність нитки I"); a.set_title("(в) зсув 0,3, rc 1,75: сила нитки і відстань", fontsize=10, loc="left")
for q in ax[[0, 2]]: q.grid(alpha=.25); q.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("/mnt/project-files/figures/fig213_threads_compete.png", bbox_inches="tight"); print("ok")
