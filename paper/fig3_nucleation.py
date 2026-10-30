"""(173) етап А: малюнок — запал народження (111), розподіл закрученості (112), відбір за розміром (113) зі стрілами ґратки (sk=4); (183) втручання в острів."""
import os as _os; _H = _os.path.dirname(_os.path.abspath(__file__)); S = _os.environ.get("NM_DATA", _H + "/../data/")
import json, re, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys; sys.path.insert(0, _H); from pstyle import V1
R = S + "r173/ana/"
b = json.load(open(R + "birth111.json"))["res"]; rows = json.load(open(R + "sel113.json"))
log = open(R + "bist112.log").read(); marg = np.array(eval(re.search(r"marginal \|hol\| \(bins of 0.0785\): (\[.*?\])", log).group(1)))
fig, ax = plt.subplots(2, 2, figsize=(11, 7.6)); ax = ax.ravel()
lags = sorted(b, key=int); x = [int(l) for l in lags]
for q, lab, c in (("hol", "|arg P|", "C0"), ("A2", "|z|² / 10", "C1"), ("phsync", "phase coherence", "C2"), ("psi", "ψ", "C3")):
    k = 10 if q == "A2" else 1
    ax[0].plot(x, [b[l][q]["birth"]/k for l in lags], "o-", c=c, label=lab + " (birth site)")
    ax[0].plot(x, [b[l][q]["ctrl"]/k for l in lags], "s--", c=c, alpha=.5)
ax[0].set_xlabel("time before nucleation"); ax[0].set_title("(a) trigger: birth site (solid) vs Matrix (dashed)"); ax[0].legend(fontsize=7, loc="lower left", bbox_to_anchor=(0, 0.12)); ax[0].invert_xaxis()
ax[1].bar(np.arange(len(marg))*0.0785 + 0.039, marg, width=0.075, color="0.4"); ax[1].set_xlabel("|arg P| (Matrix nodes)"); ax[1].set_ylabel("%"); ax[1].set_title("(b) two states: flat ≈0.2, boiling ≈1.1")
s_b = [r["size"] for r in rows if r["born"]]; s_n = [r["size"] for r in rows if not r["born"]]
bins = np.logspace(np.log10(3), np.log10(5000), 30)
ax[2].hist(s_n, bins, histtype="step", lw=2, label="did not nucleate", weights=np.full(len(s_n), 1/len(s_n))); ax[2].hist(s_b, bins, histtype="step", lw=2, label="nucleated a domain", weights=np.full(len(s_b), 1/len(s_b))); ax[2].set_ylabel("fraction of patches in class")
ax[2].set_xscale("log"); ax[2].set_xlabel("flat patch size (nodes)"); ax[2].set_title("(c) selection by size (min 73, AUC 0.97)"); ax[2].legend(fontsize=8)
rw = json.load(open(S + "r183/ana183.json")); V = [("ctl", "unchanged", V1), ("pn", "phase noise\n0.5 rad", "#e1a03a"), ("scr", "random\nphases", "#c4572e"), ("sham", "random phases\nelsewhere", "0.55")]
for i, (v, lab, c) in enumerate(V):
    nb = sum(r[v] is not None for r in rw); dl = [r[v] - r["t_j"] for r in rw if r[v] is not None]
    ax[3].bar(i, nb, color=c); ax[3].text(i, nb + 0.8, f"{nb}/{len(rw)}\n{np.median(dl):+.0f}", ha="center", va="bottom", fontsize=8)
ax[3].set_xticks(range(4)); ax[3].set_xticklabels([v[1] for v in V], fontsize=8); ax[3].set_ylim(0, 52); ax[3].set_ylabel("domains born at the site")
ax[3].set_title("(d) intervention 3 units before nucleation (disk R = 6)")
plt.tight_layout(); import sys; sys.path.insert(0, _H); from pstyle import save; save(fig, "fig3_nucleation")
