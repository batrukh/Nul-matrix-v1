"""Fig. 11 (v1.3): exchange of two vortices with a live director. (a) bound odd modes at phi = 0; (b) overlap of neighbouring steps;
(c) exchange phases. Data: data/r320 (produced by src/ext/r320: relax.py, hess.py, braid.py; effective model src/ext/r313/eff.py)."""
import os, sys, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
from pstyle import plt, tag, save
D = H + "/../data/r320/"
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
z = np.load(D + "bound_k0.npz"); c = z["c"]; v1, v2 = z["v1"], z["v2"]
a = ax[0]; a.scatter(z["x"].ravel(), z["y"].ravel(), c=z["rho"].ravel(), s=14, cmap="magma_r", vmin=0, vmax=1)
th = np.linspace(0, np.pi, 50); a.plot(c[0] + 8*np.cos(th), c[1] + 8*np.sin(th), "c--", lw=1); a.plot([v1[0], v2[0]], [v1[1], v2[1]], "b-", lw=1)
a.set_xlim(c[0] - 16, c[0] + 16); a.set_ylim(c[1] - 12, c[1] + 12); a.set_aspect("equal"); a.set_title("bound odd modes at φ = 0", fontsize=10); tag(a, "a")
b48 = json.load(open(D + "braid_N48_k.json")); b24 = json.load(open(D + "braid_N24_k.json")); a = ax[1]
a.plot(np.arange(1, 49)/48, b48["smins"], "o-", ms=3, label="step π/48"); a.plot(np.arange(1, 25)/24, b24["smins"], "s-", ms=3, label="step π/24")
a.set_ylim(0, 1.02); a.set_xlabel("φ / π"); a.set_ylabel("min. overlap of neighbouring steps"); a.legend(fontsize=8); a.set_title("adiabatic transport", fontsize=10); tag(a, "b")
eff = dict(two=[-0.9859, 0.0, 0.9859, 1.0], one=[-0.5115, -0.4974, 0.4974, 0.5115])          # src/ext/r313/eff.py, L = 48
xs = {"effective": (eff["one"], eff["two"]), "full, π/24": (b24["phases1"], b24["two_one_per_vortex"]), "full, π/48": (b48["phases1"], b48["two_one_per_vortex"])}
a = ax[2]
for i, (lab, (o, t)) in enumerate(xs.items()):
    a.scatter(np.full(4, i) - 0.12, o, marker="o", c="#2a7fd4", label="one quantum" if i == 0 else None)
    a.scatter(np.full(4, i) + 0.12, t, marker="^", c="#d4572a", label="two quanta, one per vortex" if i == 0 else None)
a.scatter([1.3, 2.3], [b24["AA_phase"], b48["AA_phase"]], marker="*", s=120, c="k", label="ground state $A_LA_R$")
for y in (-1, -.5, 0, .5, 1): a.axhline(y, color="grey", lw=.4, ls=":")
a.set_xticks(range(3)); a.set_xticklabels(list(xs)); a.set_ylabel("exchange phase / π"); a.legend(fontsize=7, loc="center left")
a.set_title("exchange phases (±1: fermion)", fontsize=10); tag(a, "c")
plt.tight_layout(); save(fig, "fig11_exchange")
