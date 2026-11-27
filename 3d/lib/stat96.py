"""stat96: чесна статистика (96) — густина світів за L і поріг безладу з нахилом τ (базова модель: iso, ℓ=8, D=1, g=0,02, запас 3,5, n=100).
Світ — компонента ≥ 20 вузлів наприкінці; кипіння — середнє 'boil' за останні 50 одиниць часу."""
import json, glob, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def one(f, L):
    d = json.load(open(f)); fin = [x for x in d["final"] if x >= 20]; w = d["w2"]; T = w[-1]["t"]
    b = np.mean([q["boil"] for q in w if "boil" in q and q["t"] > T - 50])
    return len(fin), sum(fin)/L/L, b
out = {}
for L, ss in ((96, range(1, 9)), (144, range(1, 7))):
    out[f"L{L}"] = [one(f"lc_s03.5_n100_L{L}_s{s}_tau8_D1_g0.02_iso.json", L) for s in ss]
out["reg_s1"] = [one("lc_s03.5_n100_L96_s1_tau8_D1_g0.02_iso.json", 96)]
for r, p in (("0.3", 0), ("0.45", 3), ("0.5", 7)):
    out[f"rnd{r}"] = [one(f, 96) for f in sorted(glob.glob(f"lc_s03.5_n100_L96_s1_tau8_D1_g0.02_rnd{r}g*_iso.json"))]
S = {}
for k, v in out.items():
    a = np.array(v); n = len(a); se = a.std(0, ddof=1)/np.sqrt(n) if n > 1 else np.zeros(3)
    S[k] = dict(n=n, worlds=[a[:, 0].mean(), se[0]], frac=[a[:, 1].mean(), se[1]], boil=[a[:, 2].mean(), se[2]], with_world=int((a[:, 0] > 0).sum()), raw=a.tolist())
    print(k, n, "worlds %.2f±%.2f" % tuple(S[k]["worlds"]), "frac %.3f±%.3f" % tuple(S[k]["frac"]), "boil %.3f±%.3f" % tuple(S[k]["boil"]), "with world", S[k]["with_world"])
json.dump(S, open("stat96.json", "w"), indent=1)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for i, L in enumerate((96, 144)):
    a = np.array(out[f"L{L}"]); d = a[:, 0]/L/L*1000
    ax[0].scatter(np.full(len(d), i) + np.linspace(-.12, .12, len(d)), d, color="#4a7bb7", alpha=.6)
    ax[0].errorbar(i + .3, d.mean(), d.std(ddof=1)/np.sqrt(len(d)), fmt="o", color="k", capsize=4)
ax[0].set_xticks([0, 1], ["L96 (8 реаліз.)", "L144 (6 реаліз.)"]); ax[0].set_xlim(-.5, 1.6)
ax[0].set_ylabel("світів на 1000 вузлів"); ax[0].set_title("а) густина світів, нахил τ")
labels = ["правильна\n(τ реаліз. 1)", "зсув вузлів\n0 %", "3 %", "7 %"]
for i, k in enumerate(("reg_s1", "rnd0.3", "rnd0.45", "rnd0.5")):
    f = np.array(out[k])[:, 1]*100
    ax[1].scatter(np.full(len(f), i) + np.linspace(-.1, .1, len(f)), f, color="#c0623b", alpha=.7)
    if len(f) > 1: ax[1].errorbar(i + .25, f.mean(), f.std(ddof=1)/np.sqrt(len(f)), fmt="o", color="k", capsize=4)
ax[1].set_xticks(range(4), labels); ax[1].set_ylabel("частка вузлів у світах, %")
ax[1].set_title("б) безлад сітки (неправильні вузли), L96")
plt.tight_layout(); plt.savefig("/mnt/project-files/figures/fig10_stats.png", dpi=130)
