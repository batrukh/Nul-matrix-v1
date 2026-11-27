import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, sys
out = sys.argv[1]
arms = [("s2.9_L16_mu0.2", "$s_0$=2.9, L16"), ("s3_L16_mu0.2", "$s_0$=3.0, L16"), ("s3.15_L16_mu0.2", "$s_0$=3.15, L16"), ("s3_L16_mu-0.2", "$s_0$=3.0, L16, μ=−0.2"), ("s2.9_L24_mu0.2", "$s_0$=2.9, L24")]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
cols = ["C0", "C1", "C2", "C3", "C4"]
for (k, lab), c in zip(arms, cols):
    d = json.load(open(f"lin_{k}_s1.json")); a = np.array([x[:3] for x in d["cmp"]]); r, dat, lin = a.T
    sg = np.sign(d["mu"])
    ax[0].plot(r, sg*lin, "-", color=c, lw=1.2); ax[0].plot(r, sg*dat, "o", color=c, ms=3.5, label=lab)
    m = r >= 1; w = (r >= 2) & (r <= d["L"]/2-2); kk = np.sum(dat[w]*lin[w])/np.sum(lin[w]**2); _ = np.sum(dat[m]*lin[m])/np.sum(lin[m]**2)
    ax[1].plot(r[m], (dat[m]-kk*lin[m])/abs(lin[r == 2][0])*100, "o-", color=c, ms=3, lw=0.8, label=f"{lab}: data/theory {kk:.2f}")
ax[0].axhline(0, color="k", lw=0.5); ax[0].set_yscale("symlog", linthresh=1e-3)
ax[0].set_xlabel("distance from source r (lattice units)"); ax[0].set_ylabel("sign(μ)·δρ(r) (relative to box mean)")
ax[0].set_title("points: runs (T=120); lines: lattice linear response\n(no free parameters)", fontsize=10); ax[0].legend(fontsize=8)
ax[1].axhline(0, color="k", lw=0.5); ax[1].set_ylim(-3, 3)
ax[1].set_xlabel("r"); ax[1].set_ylabel("(data − A·theory) / δρ(2), %")
ax[1].set_title("shape deviation after one amplitude A\n(A fitted on 2 ≤ r ≤ L/2−2)", fontsize=10); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(out, dpi=130)
