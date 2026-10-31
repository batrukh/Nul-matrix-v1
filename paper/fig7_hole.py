"""Fig. 7 of the paper: a hole in the v1 substrate seeding a domain (177). (a) map R × q at s0 = 4.5; (b) ψ near the hole; (c) radial intensity profile; (d) inside the window s0 = 3.5."""
import sys, json, glob, numpy as np; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__))); from pstyle import *
import os as _os; _H = _os.path.dirname(_os.path.abspath(__file__)); D_ = _os.environ.get("NM_DATA", _H + "/../data/")
sys.path.insert(0, _H + "/../src"); import ana177 as A
S = D_ + "r177/"
fig, ax = plt.subplots(2, 2, figsize=(11, 7.6)); ax = ax.ravel()
a = ax[0]; tag(a, "a"); M = json.load(open(S + "main/ana177.json"))
Rs, qs = [2, 4, 6, 8, 10, 12], [0.05, 0.1, 0.15, 0.3, 0.6, 0.9]
for r in M:
    if "s4.5_" not in r["run"] or r["R"] == 0: continue
    k = int(r["run"].split("seed")[1][0]); x = Rs.index(int(r["R"])) + (k - 1.5)*0.25; y = qs.index(r["q"])
    st = "world" if r["final"]["n"] > 0 else ("transient" if r["t1"] is not None else "none")
    a.scatter(x, y, s=60, marker={"world": "o", "transient": "o", "none": "x"}[st], facecolor={"world": V1, "transient": "white", "none": GRAY}[st],
              edgecolor={"world": V1, "transient": V1, "none": GRAY}[st] if st != "none" else None, color=GRAY if st == "none" else None, lw=1.3)
a.scatter([], [], s=50, color=V1, label="domain persists"); a.scatter([], [], s=50, facecolor="white", edgecolor=V1, label="nucleus dies"); a.scatter([], [], s=50, marker="x", color=GRAY, label="nothing")
a.set_xticks(range(len(Rs))); a.set_xticklabels(Rs); a.set_yticks(range(len(qs))); a.set_yticklabels([f"{q:g}" for q in qs])
a.set(xlabel="hole radius R", ylabel="depth q (fraction removed)", title="quiet Matrix, s₀ = 4.5 (no hole: no domains)"); a.legend(fontsize=7, loc="lower left")
a = ax[1]; tag(a, "b"); L = 48; X, Y = tri_xy(L)
Sp = np.load(glob.glob(S + "main/s4.5_seed1_R12_q0.6/*_sp.npz")[0]); k = int(np.argmin(np.abs(Sp["t"] - 30))); p = Sp["psi"][k].reshape(L, L)
a.scatter(X.ravel(), Y.ravel(), c=p.ravel(), cmap="BrBG", vmin=-1.4, vmax=1.4, s=9, marker="h", lw=0)
c = (L//2 + L//4, np.sqrt(3)/2*(L//2)); a.add_patch(plt.Circle(c, 12, fill=False, ls="--", color=INK, lw=1))
a.set_aspect("equal"); a.axis("off"); a.set_title(f"ψ at t = {Sp['t'][k]:.0f}: domain (green) on the rim of the hole R = 12", fontsize=8)
a = ax[2]; tag(a, "c"); import stall43 as T
f = T.Sys(L=L).fab; D = A.dist(L); Dm = np.maximum(D[f.tail], D[f.head]); bins = np.arange(0, 25, 1.0)
for R, ls in ((8, "-"), (12, "--")):
    Sp = np.load(glob.glob(S + f"main/s4.5_seed1_R{R}_q0.6/*_sp.npz")[0]); k = int(np.argmin(np.abs(Sp["t"] - 15))); A2 = np.abs(Sp["z"][k])**2
    prof = [A2[(Dm >= b) & (Dm < b + 1)].mean() for b in bins]; a.plot(bins + 0.5, prof, ls=ls, color=V1, label=f"R = {R}, q = 0.6, t = 15")
a.axhspan(2.7, 4.2, color=V1, alpha=0.12, lw=0); a.text(23.5, 3.45, "budget window\nof domain birth", ha="right", va="center", fontsize=7, color=V1)
a.axhline(4.5, color=GRAY, lw=1, ls=":"); a.set(xlabel="distance from hole centre", ylabel="mean link intensity |z|²", title="hole → dead ring around a dense core;\nthe outer slope crosses the window", ylim=(0, 6.5)); a.legend(fontsize=7, loc="lower right")
a = ax[3]; tag(a, "d"); W = json.load(open(S + "win/ana177.json"))
for j, (lab, R) in enumerate((("no hole", 0), ("hole R = 8", 8))):
    v = [r["t1"] for r in W if r["R"] == R]; a.scatter(j + 0.08*(np.arange(len(v)) - (len(v) - 1)/2), v, color=V1 if R else GRAY, s=40, zorder=3)
    on = sum(1 for r in W if r["R"] == R and r["first_on"] is not None and r["first_on"] <= r["first_any"])
    a.text(j, 11.3, f"first domain\nat hole: {on}/{len(v)}", ha="center", fontsize=7)
a.set_xticks([0, 1]); a.set_xticklabels(["no hole", "hole R = 8,\nq = 0.6"]); a.set(xlim=(-0.6, 1.6), ylim=(0, 13), ylabel="time of first domain", title="in the window, s₀ = 3.5")
plt.tight_layout(); save(fig, "fig7_hole")
