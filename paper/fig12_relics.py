"""Fig. 12 (v1.3): relics of domain birth with a live director. (a) disclinations and Alice vortices in domains; (b) longest vortex life per run;
(c) new disclinations on merger seams vs growth fronts; (d) disclinations at t = 150 vs time to cover half of the final domain area.
Data: data/r321, data/r322 (runs/director.sh; analysis src/ext/r321, src/ext/r322)."""
import os, sys, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
from pstyle import plt, tag, save
from scipy.stats import spearmanr
D = H + "/../data/"
fig, ax = plt.subplots(1, 4, figsize=(18, 4.4)); a = ax[0]; runs = {}
for s, c in (("1", "C0"), ("2", "C1")):
    dt = json.load(open(D + f"r321/dtrack_s{s}_d1.json")); t = [q["k"]*0.05 for q in dt]; runs[s] = dt
    a.plot(t, [max(q["nd"], .5) for q in dt], c, lw=2, label=f"seed {s}: director disclinations")
    a.plot(t, [max(q["nz2"], .5) for q in dt], c, ls=":", label=f"seed {s}: Alice vortex")
for i, s in enumerate("3456"):
    dt = json.load(open(D + f"r322/dtrack_s{s}.json")); runs[s] = dt
    a.plot([q["k"]*0.05 for q in dt], [max(q["nd"], .5) for q in dt], color=f"C{i+2}", lw=1, alpha=.7, label=f"seed {s} (T = 300)")
a.set(yscale="log", xlabel="time t", ylabel="defect triangles in domains"); a.legend(fontsize=7); a.set_title("disclinations freeze", fontsize=10); tag(a, "a")
L = json.load(open(D + "r322/lives.json")); a = ax[1]
a.bar(range(8), L["life"], color=["grey"]*2 + ["C0"]*6); a.set_xticks(range(8)); a.set_xticklabels(L["labels"], rotation=45, fontsize=8)
a.axhline(max(L["life"][:2]), color="grey", ls="--"); a.set_ylabel("longest Alice-vortex lifetime"); a.set_title("longest vortex life per run", fontsize=10); tag(a, "b")
o = json.load(open(D + "r322/ana322b.json"))["out"]; a = ax[2]
a.bar([0, 1], [1000*o["seam"][0]/o["seam"][1], 1000*o["front"][0]/o["front"][1]], color=["C3", "C7"])
for i, k in enumerate(["seam", "front"]): a.text(i, 1000*o[k][0]/o[k][1] + 2, f"{o[k][0]} new\nalive after 50: {100*o[k][3]/o[k][0]:.0f}%", ha="center", fontsize=8)
a.set_xticks([0, 1]); a.set_xticklabels(["merger seam", "ordinary growth front"]); a.set_ylim(0, 115)
a.set_ylabel("new disclinations per 1000 new domain nodes"); a.set_title("scars of domain mergers", fontsize=10); tag(a, "c")
th, n150 = [], []
for s, dt in runs.items():
    t = np.array([q["k"]*0.05 for q in dt]); aw = np.array([q["aw"] for q in dt]); nd = np.array([q["nd"] for q in dt])
    Af = aw[t <= 300][-1]; th.append(t[np.argmax(aw > 0.5*Af)]); n150.append(nd[np.argmin(abs(t - 150))])
r = spearmanr(th, n150); a = ax[3]; a.plot(th, n150, "o", ms=8)
for s, x, y in zip(runs, th, n150): a.annotate(s, (x + .4, y + .15))
a.set_xlabel("time to cover half of final domain area"); a.set_ylabel("disclinations in domains at t = 150")
a.set_title(f"slower birth, more relics (Spearman {r[0]:.2f}, p = {r[1]:.2f})", fontsize=10); tag(a, "d")
plt.tight_layout(); save(fig, "fig12_relics")
