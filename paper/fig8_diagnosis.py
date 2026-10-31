"""Fig. 8 of the paper: diagnoses of the negative results.
(a) Nordtvedt effect (155): inertial mass of a vortex in its own well of depth K, m_I/m = (a/g)₀/(a/g)_K from ff155 (r155), versus ⟨1/N³⟩ (numbers of journal entry (155)).
(b) 3D (170): largest flat 3D islands (r170/ana170.json: size, mean ψ) versus flat 2D v1 patches before nucleation and without it (r173 sel113)."""
import os as _os; _H = _os.path.dirname(_os.path.abspath(__file__)); S = _os.environ.get("NM_DATA", _H + "/../data/")
import sys, json, numpy as np; sys.path.insert(0, _H); from pstyle import *
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.7))
a = ax[0]; tag(a, "a")
acc = {K: json.load(open(S + f"r155/ff155_K{K}_g0.006.json"))["acc"] for K in (0, 1, 4, 8)}
mI = {K: acc[0]/acc[K] for K in (1, 4, 8)}; pred = {8: 1.08, 4: 1.16, 1: 1.99}            # ⟨1/N³⟩ from journal entry (155)
Ks = [8, 4, 1]; a.plot([pred[K] for K in Ks], [mI[K] for K in Ks], "o", color=V1, ms=7)
for K in Ks: a.annotate(f"K = {K}", (pred[K], mI[K]), textcoords="offset points", xytext=(6, -10), fontsize=8)
x = np.linspace(1, 2.5, 10); a.plot(x, x, "k--", lw=1, label="m_I/m = ⟨1/N³⟩"); a.axhline(1, color=GRAY, lw=1, label="equivalence principle")
a.set(xlabel="⟨1/N³⟩ over the vortex", ylabel="inertial mass m_I / m", title="scalar metric: vortex gets\nheavier in its own well"); a.legend(fontsize=7, loc="upper left")
a = ax[1]; tag(a, "b"); rows = json.load(open(S + "r173/ana/sel113.json"))
nb = [r for r in rows if not r["born"] and r["size"] >= 20]; bo = [r for r in rows if r["born"]]
a.plot([r["size"] for r in nb], [r["psi"] for r in nb], ".", color=GRAY, ms=3, alpha=0.5, label="2D, flat patch, no domain")
a.plot([r["size"] for r in bo], [r["psi"] for r in bo], "o", color=V1, ms=4, label="2D, flat patch before nucleation")
for d in json.load(open(S + "r170/ana170.json")):
    a.plot(d["top"][0], d["psi_big"], "s", color="#c4572e", ms=7, mec="k", mew=0.4)
a.plot([], [], "s", color="#c4572e", label="3D, largest island (L16, L24)")
a.set(xscale="log", xlabel="island size (nodes)", ylabel="mean ψ in island", title="3D islands do not raise ψ\n(2D: lattice arrows; 3D: τ arrows)"); a.legend(fontsize=7, loc="lower right")
plt.tight_layout(); save(fig, "fig8_diagnosis"); print(mI)
