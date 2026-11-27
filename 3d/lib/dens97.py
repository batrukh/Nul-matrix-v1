"""dens97: № 67 крок 5 — густина світів L96 (×8), L144 (×6), L192 (×4); світ ≥ 20 вузлів, ψ > 0."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = {}
for L, ss in ((96, range(1, 9)), (144, range(1, 7)), (192, range(1, 5))):
    a = []
    for s in ss:
        d = json.load(open(f"lc_s03.5_n100_L{L}_s{s}_tau8_D1_g0.02_iso.json")); fin = [x for x in d["final"] if x >= 20]
        w = d["w2"]; T = w[-1]["t"]; b = np.mean([q["boil"] for q in w if "boil" in q and q["t"] > T - 50])
        a.append((len(fin)/L/L*1000, sum(fin)/L/L, b, max(fin) if fin else 0)); print(L, s, sorted(fin, reverse=True), "кип %.3f" % b)
    a = np.array(a); n = len(a); R[L] = dict(n=n, mean=a.mean(0).tolist(), se=(a.std(0, ddof=1)/np.sqrt(n)).tolist(), raw=a.tolist())
    print(L, "густина %.3f±%.3f  частка %.4f±%.4f  кипіння %.3f±%.3f" % (R[L]["mean"][0], R[L]["se"][0], R[L]["mean"][1], R[L]["se"][1], R[L]["mean"][2], R[L]["se"][2]))
json.dump(R, open("dens97.json", "w"), indent=1)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for j, (lab, sc) in enumerate((("світів на 1000 вузлів", 1), ("частка вузлів у світах, %", 100))):
    for i, L in enumerate((96, 144, 192)):
        v = np.array(R[L]["raw"])[:, j]*sc
        ax[j].scatter(np.full(len(v), i) + np.linspace(-.12, .12, len(v)), v, color="#4a7bb7", alpha=.6)
        ax[j].errorbar(i + .3, v.mean(), v.std(ddof=1)/np.sqrt(len(v)), fmt="o", color="k", capsize=4)
    ax[j].set_xticks(range(3), [f"L{L} (×{R[L]['n']})" for L in (96, 144, 192)]); ax[j].set_ylabel(lab); ax[j].set_ylim(bottom=0)
ax[0].set_title("а) густина світів не залежить від розміру"); ax[1].set_title("б) частка Матриці у світах")
plt.tight_layout(); plt.savefig("/mnt/project-files/figures/fig11_size.png", dpi=130)
