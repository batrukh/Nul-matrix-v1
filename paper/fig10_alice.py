"""Fig. 10 (v1.2): the Z2 vortex of the variant base is an Alice string. (a) synthetic world, signed photon field behind the vortex;
(b) synthetic world, sign coefficient C behind the vortex vs the Aharonov-Bohm alpha=1/2 solution; (c) real world (seed 2) C map;
(d) real worlds (seeds 1, 2) C profiles vs AB. Data: data/r285, data/r286 (produced by src/abw285.py, src/ana285.py, src/realw286.py, src/ana286.py)."""
import os, sys, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H); sys.path.insert(0, H + "/../src"); sys.path.insert(0, H + "/../src/variant")
from pstyle import plt, tag, save, INK
import abtheory as A, matter98 as M98, phi90 as PH
D = H + "/../data/"
fig, ax = plt.subplots(2, 2, figsize=(11, 9))
# (a) synthetic snapshot
Z = np.load(D + "r285/snap_lam8_t72.npz"); X, Y = Z["X"], Z["Y"]; s = Z["vort"]; m = np.abs(s).max()
a = ax[0, 0]; a.scatter(X.ravel(), Y.ravel(), c=s.ravel(), s=3, cmap="RdBu_r", vmin=-m, vmax=m, rasterized=True)
a.plot([80], [55.43], "*", color="gold", mec="k", ms=14); a.plot([80.25, 80.25], [55.43, 90.07], "k:", lw=1); a.plot([80], [90.07], "*", color="gold", mec="k", ms=9)
a.annotate("", xy=(58, 28), xytext=(44, 28), arrowprops=dict(arrowstyle="->", color=INK)); a.text(44, 31, "incoming wave", fontsize=9)
a.set_xlim(40, 140); a.set_ylim(20, 95); a.set_aspect("equal"); a.set_xlabel("x"); a.set_ylabel("y")
a.set_title("synthetic world, λ = 8, t = 72: signed photon field", fontsize=10); tag(a, "a")
# (b) synthetic profiles
R = json.load(open(D + "r285/ana285.json")); a = ax[0, 1]
for d, col in (("12", "C0"), ("20", "C3")):
    rows = R["8"]["rows"][d]; dy = [q["dy"] for q in rows]
    a.plot(dy, [q["C"] for q in rows], "o", color=col, ms=4, label=f"model C, d = {d}")
    a.plot(dy, [q["ab_re"] for q in rows], "-", color=col, lw=1.2, label=f"AB α = ½, d = {d}")
    a.plot(dy, [q["R"] for q in rows], "s", color=col, ms=3, mfc="none", label=f"model energy ratio, d = {d}")
a.axhline(0, color="0.6", lw=0.5); a.set_ylim(-1.6, 1.8); a.set_xlabel("offset across the axis behind the vortex (nodes)"); a.set_ylabel("C,  energy ratio")
a.set_title("synthetic world, λ = 8: sign flips across the axis", fontsize=10); a.legend(fontsize=7, ncol=1, loc="lower left"); tag(a, "b")
# (c) real world, seed 2
J = json.load(open(D + "r286/rw_s2_lam6.json")); Q = np.load(D + "r286/rw_s2_lam6.npz"); X, Y = PH.coords(96)
C = Q["num"]/np.maximum(Q["den"], 1e-30); den = Q["den"]; vx, vy = J["v"]; ax_ = np.array(J["ax"]); pe = np.array(J["pe"])
r, dx, dy = M98.wrapd(X, Y, vx, vy, 96); par = dx*ax_[0] + dy*ax_[1]; per = dx*pe[0] + dy*pe[1]
w = r < 24; mw = w & (den > 0.02*den.max()); a = ax[1, 0]
a.scatter(par[w], per[w], c=Q["psi"][w], cmap="Greys_r", vmin=-1, vmax=1, s=14, marker="h", rasterized=True)
a.scatter(par[mw], per[mw], c=C[mw], cmap="RdBu_r", vmin=-1.3, vmax=1.3, s=14, marker="h", rasterized=True)
a.plot(0, 0, "*", color="gold", mec="k", ms=14); a.plot([0, 0], [0, J["s_end"]], "k:", lw=1)
a.annotate("", xy=(-8, -18), xytext=(-20, -18), arrowprops=dict(arrowstyle="->", color=INK))
a.set_xlim(-22, 22); a.set_ylim(-20, 20); a.set_aspect("equal"); a.set_xlabel("along the beam from the vortex (nodes)"); a.set_ylabel("across the beam")
a.set_title("real world (P = 0.16, seed 2, λ = 6): C, red +, blue −", fontsize=10); tag(a, "c")
# (d) real profiles
G = json.load(open(D + "r286/ana286.json")); a = ax[1, 1]
for sd, col in ((1, "C0"), (2, "C1")):
    for d, mk in ((6, "o"), (8, "s")):
        rr = [q for q in G[f"rw_s{sd}_lam6"]["rows"] if q["d"] == d and q["dist"] >= 1.5]
        a.plot([q["y"] for q in rr], [q["C"] for q in rr], mk, color=col, ms=4, label=f"seed {sd}, d = {d}")
yy = np.arange(-10, 10.5, 0.5); k = 2*np.pi/6.0; th = A.ab_sim(np.full_like(yy, 7.0), yy, k)/np.exp(1j*k*7.0)
a.plot(yy, th.real, "k-", lw=1.5, label="AB α = ½, free space, d = 7"); a.axhline(0, color="0.6", lw=0.5)
a.set_ylim(-1.8, 1.8); a.set_xlabel("offset across the axis behind the vortex (nodes)"); a.set_ylabel("C")
a.set_title("real worlds, λ = 6: sign flips as in AB", fontsize=10); a.legend(fontsize=7, loc="lower left"); tag(a, "d")
plt.tight_layout(); save(fig, "fig10_alice")
