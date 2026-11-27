"""thr97: № 67 крок 4 — чи залежать числа світів від порогу ψ (ψ > 0 проти бар'єра ψ > −0,34 і ψ > 0,5), на готових станах (96)."""
import numpy as np, json, worlds90 as W
def count(psi, thr):
    ws, _, _ = W.worlds(psi - thr); ws = [w["n"] for w in ws if w["n"] >= 20]
    return len(ws), sum(ws)/psi.size
out = {}
for L, ss in ((96, range(1, 9)), (144, range(1, 7))):
    for s in ss:
        psi = np.load(f"lc_s03.5_n100_L{L}_s{s}_tau8_D1_g0.02_iso.npz")["psi"]; psi = psi.reshape(L, L)
        r = {str(t): count(psi, t) for t in (-0.34, 0.0, 0.5)}
        mid = float(((psi > -0.34) & (psi <= 0)).mean())
        out[f"L{L}_s{s}"] = dict(r, mid=mid, psimin=float(psi.min()), psimax=float(psi.max())); print(L, s, r, "між порогами %.4f" % mid, flush=True)
json.dump(out, open("thr97.json", "w"), indent=1)
for L in (96, 144):
    for t in ("-0.34", "0.0", "0.5"):
        a = np.array([v[t] for k, v in out.items() if k.startswith(f"L{L}_")])
        print(L, t, "світів %.2f±%.2f" % (a[:, 0].mean(), a[:, 0].std(ddof=1)/np.sqrt(len(a))), "частка %.4f±%.4f" % (a[:, 1].mean(), a[:, 1].std(ddof=1)/np.sqrt(len(a))))
