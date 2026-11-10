# fig212: (317) нитки обирають сусідів — хаби замість порядку; порядок лише з членом кривини руками
import json, ast, sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/mnt/project-files/model/s52/r317"); import graph317 as G
R = [("avg_lam0_L50", "бюджет порівну, I середнє", "#eb6834"), ("geo_lam0_L50", "бюджет порівну, I геометр.", "#f2a33a"),
     ("geo_lam0.003_L50", "те саме + кривина λ 0,003 (руками)", "#3987e5"), ("zero_lam1_L50", "лише кривина λ (руками)", "#0d366b")]
def logs(n): return [ast.literal_eval(l.replace("np.float64(", "(")) for l in open(f"{n}.log") if l.startswith("{")]
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), dpi=150)
for n, lab, c in R:
    L = logs(n); ax[0].plot([q["t"] for q in L], [100*q["f6"] for q in L], color=c, lw=2, label=lab)
    k = np.array(json.load(open(f"{n}.json"))["k"]); b = np.bincount(k, minlength=70)
    ax[1].semilogy(np.arange(3, 70), np.where(b[3:70] > 0, b[3:70]/len(k), np.nan), "o-", color=c, ms=3, lw=1.2)
    D = json.load(open(f"{n}.json")); s = G.Tri.__new__(G.Tri); s.n = len(k); s.k = k; s.adj = [set() for _ in range(s.n)]
    for a, bb in D["edges"]: s.adj[a].add(bb); s.adj[bb].add(a)
    g = G.grains(s)
    if g: ax[2].plot(np.arange(1, len(g) + 1), g, "o-", color=c, ms=3, lw=1.2)
ax[0].set_xlabel("прохід Монте-Карло (охолодження)"); ax[0].set_ylabel("вузлів із 6 сусідами, %"); ax[0].set_title("(а) порядок під час охолодження, N 2500", fontsize=10, loc="left"); ax[0].legend(fontsize=7.5, frameon=False)
ax[1].set_xlabel("сусідів у вузла k"); ax[1].set_ylabel("частка вузлів"); ax[1].set_title("(б) розподіл сусідів у кінці", fontsize=10, loc="left")
ax[2].set_xlabel("номер зерна"); ax[2].set_ylabel("вузлів у зерні"); ax[2].set_yscale("log"); ax[2].set_title("(в) зерна порядку (лише з кривиною)", fontsize=10, loc="left")
ax[2].axhline(3200, color="#888", ls=":", lw=1.5); ax[2].text(1.2, 2300, "потрібно для світу (316): ≈3200", fontsize=7.5, color="#555")
for a in ax: a.grid(alpha=.25); a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("/mnt/project-files/figures/fig212_threads_choose_neighbours.png", bbox_inches="tight"); print("ok")
