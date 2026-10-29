"""(173) етап А: малюнок — запал народження (111), розподіл закрученості (112), відбір за розміром (113) зі стрілами ґратки (sk=4)."""
import os as _os; _H = _os.path.dirname(_os.path.abspath(__file__)); S = _os.environ.get("NM_DATA", _H + "/../data/")
import json, re, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = S + "r173/ana/"
b = json.load(open(R + "birth111.json"))["res"]; rows = json.load(open(R + "sel113.json"))
log = open(R + "bist112.log").read(); marg = np.array(eval(re.search(r"marginal \|hol\| \(bins of 0.0785\): (\[.*?\])", log).group(1)))
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
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
plt.tight_layout(); import sys; sys.path.insert(0, _H); from pstyle import save; save(fig, "fig3_nucleation")
