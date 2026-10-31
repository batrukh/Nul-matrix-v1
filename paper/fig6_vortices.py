"""Fig. 6 of the paper: Z₂ vortices in model v1. (a) number of vortices in time (174, lattice arrows) versus (166, τ arrows); (b) summary: gas at t ≥ 150 and lifetime of a planted vortex."""
import os as _os; _H = _os.path.dirname(_os.path.abspath(__file__)); S = _os.environ.get("NM_DATA", _H + "/../data/")
import sys, json, glob, numpy as np; sys.path.insert(0, _H); from pstyle import *
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
a = ax[0]; tag(a, "a"); col = {"1.3": "#c4572e", "1.5": V1}
for run, ls in (("r174", "-"), ("r166", ":")):
    for d in sorted(glob.glob(S + run + ("/b*_s*" if run == "r174" else "/data/b*_s*"))):
        b = d.split("/b")[-1].split("_")[0]; tr = json.load(open(glob.glob(d + "/*_g104.json")[0]))["track"]
        t = np.array([r["t"] for r in tr]); n = np.array([r["ncl"] for r in tr], float)
        k = 5; ns = np.convolve(n, np.ones(k)/k, "same")
        a.plot(t, ns, color=col[b] if run == "r174" else "#999", ls=ls, lw=1.3 if run == "r174" else 1.0)
for b in col: a.plot([], [], color=col[b], label=f"β = {b}, lattice arrows (model v1)")
a.plot([], [], color="#999", ls=":", label="random-field (τ) arrows, for comparison")
a.set(xlabel="t", ylabel="number of Z₂ vortices in domains", title="vortices are born with the domain, then annihilate"); a.legend(fontsize=7)
a = ax[1]; tag(a, "b"); A = json.load(open(S + "r174/ana174.json")); B = json.load(open(S + "r166/ana166.json"))
L4 = json.load(open(S + "r174/life174.json")); L6 = json.load(open(S + "r166/life166.json"))
x = np.arange(2); w = 0.35
for j, (D, lab, c) in enumerate(((B, "τ arrows", "#999"), (A, "lattice arrows (v1)", V1))):
    v = [np.mean([D[f"ela_b{b}_s{s}"]["ncl"] for s in (1, 2)]) for b in ("1.3", "1.5")]
    a.bar(x + (j - 0.5)*w, v, w, color=c, label=lab)
a.set_xticks(x); a.set_xticklabels(["β = 1.3", "β = 1.5"]); a.set_ylabel("vortices in domains at t ≥ 150 (mean)")
l4 = [L4[f"p{b}_s{s}"]["life"] for b in ("1.3", "1.5") for s in (1, 2)]; l6 = [L6[f"ela_p{b}_s{s}"]["life"] for b in ("1.3", "1.5") for s in (1, 2)]
a.set_title(f"planted vortex lives {min(l4):.0f}–{max(l4):.0f} (τ: {min(l6):.0f}–{max(l6):.0f}) time units", fontsize=9); a.legend(fontsize=7)
plt.tight_layout(); save(fig, "fig6_vortices")
