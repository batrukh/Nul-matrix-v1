"""Fig. 4 статті: злиття без пружності проти співіснування в моделі v1.
(a) число світів у часі, L96: без пружності r69 (s3, s4, T250) проти v1 ela/L96_l8_g0.01_s3,s4 (T600);
(b) частка найбільшого світу, L72: без пружності і v1 при різних P (ela/map, ℓ = 8);
(c) карта (P, ℓ): ріст найбільшого світу t 100–150 (колір) і число світів (ela/map);
(d) густина світів і медіана розміру проти L (ela/sizeL/sizeL.json, v1)."""
import os as _os; _H = _os.path.dirname(_os.path.abspath(__file__)); S = _os.environ.get("NM_DATA", _H + "/../data/")
import sys, json, ast, numpy as np; sys.path.insert(0, _H); from pstyle import *
def load(fn): s = open(fn).read(); i = s.index('{'); return ast.literal_eval(s[i:s.rindex('}') + 1])
fig, ax = plt.subplots(2, 2, figsize=(11, 7.6)); ax = ax.ravel()
a = ax[0]; tag(a, "a")
for sd, ls in ((3, "-"), (4, "--")):
    w = json.load(open(S + f"r69/lc_s03.5_n100_L96_s{sd}_sk4l0.json"))["w2"]; a.plot([q["t"] for q in w], [q["nuc"] for q in w], color=NOEL, ls=ls, lw=1.2)
    w = load(S + f"ela/L96_l8_g0.01_s{sd}/log")["w2"] + load(S + f"ela/L96_l8_g0.01_s{sd}/log600")["w2"]
    a.plot([q["t"] for q in w], [q["nuc"] for q in w], color=V1, ls=ls, lw=1.2)
a.plot([], [], color=NOEL, label="without elasticity"); a.plot([], [], color=V1, label="model v1 (P = 0.64)")
a.set(xlabel="t", ylabel="number of domains (≥ 5 nodes)", title="L = 96, two seeds"); a.legend(fontsize=7)
a = ax[1]; tag(a, "b"); N = 72**2
w = json.load(open(S + "r69/lc_s03.5_n100_L72_s1_sk4l0.json"))["w2"][:151]; a.plot([q["t"] for q in w], [100*q["big"]/N for q in w], color=NOEL, ls="--", lw=1.5, label="without elasticity")
cm = plt.get_cmap("viridis")
for k, P in enumerate((0.15, 0.22, 0.3, 0.36, 0.4, 1.0)):
    w = load(S + f"ela/map/m_l8_P{P}/log")["w2"]; a.plot([q["t"] for q in w], [100*q["big"]/N for q in w], color=cm(k/5.5), lw=1.2, label=f"P = {P:g}")
a.set(xlabel="t", ylabel="largest domain (% of system)", title="L = 72, ℓ = 8"); a.legend(fontsize=7, ncol=2)
a = ax[2]; tag(a, "c")
for d in json.load(open(S + "ela/map/map.json")):
    big = [r[4]*N for r in d["tr"]]; g = (big[-1] - big[99])/50; nw = len([v for v in d["final"] if v >= 20])
    a.scatter(d["P"], d["l"], s=60, c=[[min(max(g, 0), 5)/5, 0.35, 1 - min(max(g, 0), 5)/5]], edgecolor="k", lw=.4, zorder=3)
    a.annotate(str(nw), (d["P"], d["l"]), textcoords="offset points", xytext=(7, 3), fontsize=7)
a.axvspan(0.22, 0.36, color=GRAY, alpha=0.3, lw=0); a.axvline(0.32, color="#555", ls=":", lw=1)
a.set(xscale="log", yscale="log", xlabel="P = γℓ²", ylabel="ℓ", title="growth of largest domain (red = growing)")
a.minorticks_off(); a.set_yticks([4, 8, 16, 24, 32]); a.set_yticklabels(["4", "8", "16", "24", "32"]); a.set_xticks([0.15, 0.3, 0.64, 1.0]); a.set_xticklabels(["0.15", "0.3", "0.64", "1"])
a = ax[3]; tag(a, "d"); rows = json.load(open(S + "ela/sizeL/sizeL.json"))
Ls = sorted(set(r["L"] for r in rows)); dens = []; med = []
for L in Ls:
    rs = [r for r in rows if r["L"] == L]; sz = [v for r in rs for v in r["final"] if v >= 20]
    dens.append(np.mean([1000*len([v for v in r["final"] if v >= 20])/L**2 for r in rs])); med.append(np.median(sz) if sz else np.nan)
    a.plot([L]*len(sz), sz, ".", color=V1, alpha=0.35, ms=4)
a.plot(Ls, med, "D-", color=V1, label="median domain size"); a.set(xlabel="L", ylabel="domain size (nodes)", yscale="log", title="model v1"); a.minorticks_off(); a.set_yticks([20, 50, 100, 200]); a.set_yticklabels(["20", "50", "100", "200"])
b = a.twinx(); b.plot(Ls, dens, "s--", color="#555", label="domains per 1000 nodes"); b.set_ylim(0, 2); b.spines["right"].set_visible(True); b.set_ylabel("domains per 1000 nodes")
a.legend(fontsize=7, loc="upper left"); b.legend(fontsize=7, loc="upper right")
plt.tight_layout(); save(fig, "fig4_coexistence")
print("dens", [round(x, 2) for x in dens], "med", med)
