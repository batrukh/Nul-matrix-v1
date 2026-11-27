"""(470) аналіз: фіт δρ(r) = A·Σ_образи e^{−r/ξ}/r + C для r ≥ 2 (Кун L16), медіани по зернах, вердикт за критерієм derive470.txt."""
import json, glob, numpy as np
from scipy.optimize import curve_fit
import sys; RMIN=float(sys.argv[1])
PRED = {2.9: 4.7, 3.0: 2.9, 3.15: 2.4}
def yuk(r, L, xi):
    s = 0
    for dx in (-1, 0, 1):          # образи вздовж радіуса: грубо, ізотропне наближення
        for k in (-1, 0, 1):
            pass
    # точніше: сума по 3D-образах точки на відстані r уздовж осі
    out = np.zeros_like(r)
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                d = np.sqrt((r + i*L)**2 + (j*L)**2 + (k*L)**2); out += np.exp(-d/xi)/d
    return out
res = {}
for f in sorted(glob.glob("runs/*.json")):
    d = json.load(open(f)); L = d["L"]; P = d["prof_rho"]
    r = np.array([float(k) for k in P]); y = np.array([v[0] for v in P.values()]); w = np.array([v[1] for v in P.values()])
    m = (r >= RMIN) & (r <= L/2)
    fn = lambda rr, A, xi, C: A*yuk(rr, L, xi) + C
    try:
        p, cv = curve_fit(fn, r[m], y[m], p0=[0.1*np.sign(d["mu"]), 2.5, d["s0"]], sigma=1/np.sqrt(w[m]), bounds=([-50, 0.3, 0], [50, 60, 10]), maxfev=20000)
    except Exception as e:
        p = [np.nan]*3
    key = (d["s0"], d["mu"], L); res.setdefault(key, []).append(dict(seed=d["seed"], A=float(p[0]), xi=float(p[1]), C=float(p[2]), d2=float(y[r == 2][0] - y[m][-1]), conv=d["hist"][-1]["in_"] - d["hist"][-3]["in_"] if len(d["hist"]) >= 3 else None))
summ = {}
for (s0, mu, LL), L_ in sorted(res.items()):
    xi = np.array([q["xi"] for q in L_]); sg = np.mean([np.sign(q["A"]) == np.sign(mu) for q in L_])
    summ[f"{s0:g}_{mu:g}_L{LL}"] = dict(n=len(L_), xi_med=float(np.median(xi)), xi_q=[float(np.percentile(xi, 25)), float(np.percentile(xi, 75))], sign_ok=float(sg), pred=PRED.get(s0),
                                  conv_max=max(abs(q["conv"] or 0) for q in L_))
    print(s0, mu, LL, summ[f"{s0:g}_{mu:g}_L{LL}"])
json.dump(dict(runs={f"{k[0]:g}_{k[1]:g}_L{k[2]}": v for k, v in res.items()}, summary=summ), open(f"ana470L_r{RMIN:g}.json","w"), indent=1)
