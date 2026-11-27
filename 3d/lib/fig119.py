import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = [json.loads(l) for l in open("tab119.txt")]   # diagp.py 3d по кінцевих станах rq (119)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
for j, (kind, base, lab, c) in enumerate((("lc", 1.3, "λ_c", "C2"), ("eps0", 3.0, "ε₀", "C1"))):
    rs = [r for r in R if r["kind"] == kind]; fs = sorted({r["val"]/base for r in rs})
    mean = lambda q, f: np.mean([r[q] for r in rs if abs(r["val"]/base - f) < 1e-9])
    ax[0].plot(fs, [mean("psi", f) for f in fs], "o-", c=c, label=f"множник {lab}")
    a = ax[1 + j]
    for q, n, cc in (("coup", "зв'язок λ_c", "C2"), ("cost", "ціна ε₀", "C1"), ("grad", "градієнт κ (поверхня)", "C4"), ("pot", "потенціал", "0.5")):
        a.plot(fs, [mean(q, f) for f in fs], "o-", c=cc, label=n)
    a.axhline(0, c="k", lw=0.6); a.axhline(-0.086, c="C4", ls=":", lw=1); a.text(fs[0], -0.06, "2D: градієнт −0,09", fontsize=7, color="C4")
    a.set_xlabel(f"множник {lab}"); a.set_ylabel("сила на ψ у протосвітах"); a.set_title(f"Прохід {lab}: що росте разом із ψ"); a.legend(fontsize=7, loc="center right" if j else "upper left")
ax[0].axhline(-0.667, c="C0", ls="--"); ax[0].text(0.3, -0.655, "2D-протосвіти (−0,67)", fontsize=8, color="C0")
ax[0].axhline(-1/np.sqrt(3), c="0.5", ls=":"); ax[0].text(0.3, -0.57, "вершина бар'єра (−0,58)", fontsize=8, color="0.4")
ax[0].set_xlabel("множник сталої"); ax[0].set_ylabel("ψ у пласких ділянках"); ax[0].set_ylim(-0.9, -0.52); ax[0].legend(fontsize=8)
ax[0].set_title("3D Б, α = 1: світів 0 у всіх 12 прогонах")
plt.tight_layout(); plt.savefig("/mnt/project-files/figures/fig33_lc_eps_scan.png", dpi=130)
