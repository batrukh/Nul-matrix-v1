import json, numpy as np, phi90 as PH, matplotlib, matplotlib.ticker; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B, O, G3, INK, MUT = "#2a78d6", "#eb6834", "#9a9893", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUT, "xtick.color": MUT, "ytick.color": MUT, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4)); c = lambda x: f"{x:g}".replace(".", ",")
def E(P, b): P2 = P @ P; return -0.5*np.trace(P2) - b/3*np.trace(P2 @ P) + 0.25*np.trace(P2)**2 + 0.125*np.trace(P2 @ P2)
bs = np.linspace(0, 0.35, 15); lab = {(1, 1, -1, -1, -1): ("Φ★ (2 додатні)", B, "-"), (-1, -1, 1, 1, 1): ("−Φ★ (3 додатні)", O, "-"), (4, -1, -1, -1, -1): ("1 додатне + 4", G3, "--")}
for pat, (nm, col, ls) in lab.items():
    ys = []
    for b in bs:
        Q = np.diag(np.array(pat, float)); Q -= np.trace(Q)/5*np.eye(5); Q *= 0.5/np.linalg.norm(Q)
        for _ in range(20000): Q = Q - 0.02*PH.dV(Q[None], np.array([-1.0]), b)[0]
        ys.append(E(Q, b))
    ax[0].plot(bs, ys, ls, color=col, lw=2, label=nm)
ax[0].axvspan(0.25, 0.35, color=G3, alpha=0.15); ax[0].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: c(round(v, 2)))); ax[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: c(round(v, 2)))); ax[0].set_xlabel("b (кубічний член потенціалу Φ★)"); ax[0].set_ylabel("енергія вакууму (a = −1)")
ax[0].legend(fontsize=7, frameon=False); ax[0].set_title("а) b > 0 робить −Φ★ хибним; при b ≈ 0,3 наздоганяє вакуум «1+4»", fontsize=9)
r = json.load(open("ana123.json"))
for pb, col in (("0", O), ("0.1", B)):
    for s, ls in (("1", "-"), ("2", "--")):
        k = f"pb{pb}_b1.3_s{s}"; ax[1].plot(r[k]["t"], [100*(x if x is not None else np.nan) for x in r[k]["f3"]], ls, color=col, lw=1.6, label=f"b = {c(float(pb))}, Матриця {s}")
ax[1].set_xlabel("час"); ax[1].set_ylabel("% світу у вакуумі −Φ★"); ax[1].legend(fontsize=7, frameon=False); ax[1].set_title("б) β = 1,3: при b = 0,1 домени −Φ★ зникають", fontsize=9)
ks = [f"pb{pb}_b{be}_s{s}" for be in ("1.3", "1.5") for s in ("1", "2") for pb in ("0", "0.1")]; x = np.arange(4); w = 0.38
for j, (pb, col) in enumerate((("0", O), ("0.1", B))):
    kk = [f"pb{pb}_b{be}_s{s}" for be in ("1.3", "1.5") for s in ("1", "2")]
    ax[2].bar(x + (j - 0.5)*w, [r[k]["late"] for k in kk], w, color=col, label=f"b = {c(float(pb))}")
ax[2].set_xticks(x); ax[2].set_xticklabels(["β 1,3, М1", "β 1,3, М2", "β 1,5, М1", "β 1,5, М2"]); ax[2].set_ylabel("вихорів у середньому, t ≥ 150")
ax[2].legend(fontsize=7, frameon=False); ax[2].set_title("в) без стінок сталої речовини менше: β 1,3 — 6 замість 20", fontsize=9)
plt.tight_layout(); plt.savefig("fig37_cubic_b.png", dpi=130)
