"""Fig. 1 статті: схема моделі v1. (a) фрагмент трикутної мережі: нитки z_e зі стрілами, вузли ψ_i, петля P_t трикутника;
(b) шари моделі: Матриця (ψ_M), світ (ψ_W) з полем Φ★, розтяг полотна e(r) довкола світу (∝ ln r при r ≪ ℓ, e^(−r/ℓ) далі)."""
import os as _os; _H = _os.path.dirname(_os.path.abspath(__file__)); S = _os.environ.get("NM_DATA", _H + "/../data/")
import sys, numpy as np; sys.path.insert(0, _H); from pstyle import *
from scipy.special import k0
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})
a = ax[0]; tag(a, "a"); h = np.sqrt(3)/2
P = {(i, j): (i + j/2, j*h) for i in range(-1, 4) for j in range(0, 3)}
for (i, j), (x, y) in P.items():
    for di, dj in ((1, 0), (0, 1), (-1, 1)):
        if (i + di, j + dj) in P:
            x2, y2 = P[(i + di, j + dj)]
            a.annotate("", (x2, y2), (x, y), arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.1, shrinkA=6, shrinkB=6, mutation_scale=9))
for (x, y) in P.values(): a.plot(x, y, "o", ms=9, color="white", mec=V1, mew=1.6, zorder=3)
t = [P[(1, 0)], P[(2, 0)], P[(1, 1)]]; a.add_patch(plt.Polygon(t, color=V1, alpha=0.18, lw=0))
a.text(1.5, 0.3, r"$P_t=z_1z_2z_3$", ha="center", fontsize=8)
a.text(1.5, -0.25, r"link $z_e$ (arrow = orientation)", ha="center", fontsize=8, color="#555")
a.text(*np.add(P[(1, 1)], (0, 0.22)), r"$\psi_i$", ha="center", fontsize=9, color=V1)
a.set_aspect("equal"); a.axis("off"); a.set_title("substrate: complex links on a triangular lattice", fontsize=9)
a = ax[1]; tag(a, "b"); r = np.linspace(0, 40, 600); R = 8; ell = 8
psi = np.where(r < R, 1.125, -0.786) + 0*r; psi = -0.786 + (1.125 + 0.786)/(1 + np.exp((r - R)/0.8))
a.plot(r, psi, color=V1, lw=2, label="order parameter ψ")
e = np.where(r < R, 1.0, k0(np.maximum(r, R)/ell)/k0(R/ell)); a.plot(r, 0.9*e - 0.3, color="#c4572e", lw=1.5, ls="--", label="strain e(r) (decays on ℓ)")
phi = np.where(r < R - 1, 0.6, 0.0); a.fill_between(r, -0.95, -0.95 + phi, color="#5b3a8e", alpha=0.25, lw=0, label="matrix field Φ★ (awake only in domain)")
a.axvspan(0, R, color=V1, alpha=0.07, lw=0); a.text(R/2, 1.3, r"domain (world), $\psi>0$", ha="center", fontsize=8)
a.text(25, -0.65, r"Matrix: $\psi<0$, boiling links", ha="center", fontsize=8)
a.set(xlabel="distance from domain centre r (lattice units)", yticks=[], xlim=(0, 40), ylim=(-1, 1.6)); a.legend(fontsize=7, loc="center right")
a.set_title("schematic profile: substrate (ψ, elastic strain) and field sector (Φ★, SO(5))", fontsize=9)
plt.tight_layout(); save(fig, "fig1_model")
