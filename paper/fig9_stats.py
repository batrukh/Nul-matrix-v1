"""Fig. 4 (Statistics): budget window on 7-8 seeds, elastic arrest against P, merging without elasticity (178)."""
import json, glob, numpy as np
from pstyle import plt, V1, NOEL, INK, tag, save
import os; R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "r178")
ld = lambda p: json.load(open(glob.glob(p)[0]))
S1 = {3.0: 23.0, 3.3: 14.1, 3.5: 10.6}; B64 = {1: 6.8, 2: 3.4, 3: 2.3}          # seed 1 of the window and P = 0.64: earlier runs (r176/window176.txt)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
for s0 in (2.7, 3.0, 3.3, 3.5):
    fs = [100*json.load(open(p))["w2"][-1]["nB"]/72**2 for p in glob.glob(f"{R}/win_s{s0}_seed*.json") + glob.glob(f"{R}/lc_s0{s0:g}_n100_L72_s*_eg0.01.json")]
    if s0 in S1: fs.append(S1[s0])
    ax[0].plot([s0]*len(fs), fs, "D", color=V1, alpha=.55, ms=5); ax[0].plot(s0, np.mean(fs), "_", color=INK, ms=18, mew=2)
ax[0].set_xlabel(r"budget $s_0$"); ax[0].set_ylabel("domain fraction (%)"); tag(ax[0], "a")
Ps = [0.15, 0.22, 0.29, 0.36, 0.45, 0.64]; mk = "osv^"
for s in range(1, 5):
    y = [100*ld(f"{R}/P{P}_seed{s}.json")["w2"][-1]["big"]/72**2 for P in Ps[:-1]] + [B64.get(s, np.nan)]
    ax[1].plot(Ps, y, mk[s-1]+"-", color=V1, alpha=.4+.15*s, ms=5, label=f"seed {s}")
ax[1].set_xlabel(r"$P=\gamma\ell^2$  ($\ell=8$)"); ax[1].set_ylabel("largest domain (%)"); ax[1].legend(fontsize=9); tag(ax[1], "b")
for s in range(1, 5):
    w = ld(f"{R}/ost_g0_seed{s}.json")["w2"]; t = np.array([x["t"] for x in w]); m = np.array([x["nB"]/max(x["nuc"], 1) for x in w])
    ax[2].loglog(t[t >= 8], m[t >= 8], color=NOEL, lw=1.3, alpha=.5+.12*s)
tt = np.array([40, 300]); ax[2].loglog(tt, 200*(tt/40)**(1/3), "--", color=INK, lw=1.2, label=r"$t^{1/3}$")
ax[2].set_xlabel("time"); ax[2].set_ylabel("mean domain size (nodes)"); ax[2].legend(fontsize=9); tag(ax[2], "c")
fig.tight_layout(); save(fig, "fig9_stats")
