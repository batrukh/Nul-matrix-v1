"""Fig. 5 статті: світло в моделі v1. (a) tr Φ² на кінцевому стані (r173 B_pf1_s1, t = 200) і межі світів; (b) частка енергії «фотона»
у всіх світах і у власному світі до T = 300 (r182, 4 запуски); (c) проходження крізь межу світу (r181); (d) дисперсія: (ω/ck − 1) від ka, лінійна теорія r124 (13 напрямків) і −(ka)²/96."""
import os as _os; _H = _os.path.dirname(_os.path.abspath(__file__)); S = _os.environ.get("NM_DATA", _H + "/../data/")
import sys, json, glob, numpy as np; sys.path.insert(0, _H); from pstyle import *
fig, ax = plt.subplots(2, 2, figsize=(11, 7.6)); ax = ax.ravel()
a = ax[0]; tag(a, "a"); d = np.load(S + "r173/B_pf1_s1/lc_s03.5_n100_L96_s1_sk4l0_pf1_g91.npz"); L = 96
t2 = np.einsum("xyij,xyji->xy", d["P"], d["P"]); X, Y = tri_xy(L)
sc = a.scatter(X.ravel(), Y.ravel(), c=t2.ravel(), s=2.2, marker="h", linewidths=0, cmap="magma_r", vmin=0, vmax=0.7)
a.set_aspect("equal"); a.axis("off"); a.set_title("tr Φ★² at t = 200: the field wakes only in domains", fontsize=9)
plt.colorbar(sc, ax=a, fraction=0.035, pad=0.01)
a = ax[1]; tag(a, "b")
for f in sorted(glob.glob(S + "r182/ph182_*_T300.json")):
    q = json.load(open(f)); t, fi, fo = np.array(q["fin_t"]).T; c = V1 if q["floor"] == 0.02 else "#c4572e"
    a.plot(t, 100*fi, color=c, lw=1.2); a.plot(t, 100*fo, color=c, lw=1.0, ls=":")
a.plot([], [], color=V1, label="Matrix conductivity 0.02"); a.plot([], [], color="#c4572e", label="Matrix conductivity 0.005")
a.plot([], [], color="#555", label="in all domains"); a.plot([], [], color="#555", ls=":", label="in launching domain")
a.set(xlabel="t", ylabel="photon energy (%)", ylim=(0, 105), title="photon launched in the largest domain"); a.legend(fontsize=7, loc="upper right")
a = ax[2]; tag(a, "c"); R = [json.load(open(f)) for f in sorted(glob.glob(S + "r180/imp181_f*.json"))]
for w, c in ((0.02, V1), (0.005, "#c4572e")):
    for dl, mk in ((1.0, "o-"), (3.0, "s--")):
        q = sorted([r for r in R if r["floor"] == w and r["delta"] == dl], key=lambda r: r["kidx"])
        a.plot([r["kidx"] for r in q], [100*np.mean([1 - e["fw"] for e in r["rec"][79:100]]) for r in q], mk, color=c, ms=4,
               label=f"w = {w:g}, {'sharp' if dl == 1 else 'smooth'} boundary")
    a.axhline(400*w/(1 + w)**2, color=c, lw=.8, ls=":")
a.set(xlabel="k (per lattice step)", ylabel="transmitted into Matrix (%)", title="boundary transmission; dotted: 4w/(1+w)²"); a.legend(fontsize=7)
a = ax[3]; tag(a, "d"); r = json.load(open(S + "r124/disp124_L24.json"))["lin"]; c0 = r[0]["w"][1]/r[0]["k"]
ks = sorted(set(q["k"] for q in r if q["k"] >= 0.1))
for th in sorted(set(q["th"] for q in r)):
    pts = [(q["k"], q["w"][1]/(c0*q["k"]) - 1) for q in r if q["th"] == th and q["k"] >= 0.1]
    a.plot([p[0] for p in pts], [-p[1] for p in pts], color=V1, lw=0.8, alpha=0.6)
kk = np.linspace(0.1, 3.6, 100); a.plot(kk, kk**2/96, "k--", lw=1, label="(ka)²/96")
a.plot([], [], color=V1, label="lattice theory, 13 directions"); a.set(xscale="log", yscale="log", xlabel="ka", ylabel="1 − ω/ck", title="photon dispersion: quadratic, no linear term")
a.legend(fontsize=7)
plt.tight_layout(); save(fig, "fig5_light")
