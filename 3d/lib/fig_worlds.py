"""fig_worlds: карта Матриці — інтенсивність ниток у вузлах (кольором) і світи (ψ > 0, контур); стрілка — напрям нахилу до-часу."""
import numpy as np, sys, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import simNlib as N
fn, out = sys.argv[1], sys.argv[2]; L = 96; f = N.Fabric(L); d = np.load(fn); z, psi = d["z"], d["psi"]
I = np.abs(z)**2; In = (np.bincount(f.tail, I, f.n) + np.bincount(f.head, I, f.n))/(np.bincount(f.tail, None, f.n) + np.bincount(f.head, None, f.n))
x = f.xy[:, 0] + f.xy[:, 1]/2; y = f.xy[:, 1]*np.sqrt(3)/2; x = x % L
fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=150)
sc = ax.scatter(x, y, c=In, s=5, cmap="magma", marker="h", linewidths=0)
W = psi > 0; ax.scatter(x[W], y[W], s=7, facecolors="none", edgecolors="#4fd1c5", linewidths=0.5)
th = np.radians(-12.4); ax.annotate("", xy=(L*0.92 + 8*np.cos(th), 6 + 8*np.sin(th)), xytext=(L*0.92, 6), arrowprops=dict(arrowstyle="->", color="white", lw=1.5))
ax.text(L*0.80, 10, "нахил τ", color="white", fontsize=8)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("black")
cb = fig.colorbar(sc, ax=ax, shrink=0.8); cb.set_label("інтенсивність ниток |z|²")
ax.set_title("Матриця L96: світи (бірюзові вузли) серед киплячої Матриці", fontsize=10)
fig.tight_layout(); fig.savefig(out); print("ok", W.sum())
