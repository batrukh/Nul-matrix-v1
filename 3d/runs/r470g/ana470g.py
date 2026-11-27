"""(470Г) аналіз: дані (r470/runs) проти лінійного відгуку (lin_*.json) по плечах; + фіт A·Yuk+C до лінійного профілю тими ж вікнами, що й дані."""
import json, glob, numpy as np, sys
sys.path.insert(0, "/mnt/project-files/model/s52/r470")
from scipy.optimize import curve_fit
def yuk(r, L, xi):
    out = np.zeros_like(r)
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                d = np.sqrt((r+i*L)**2+(j*L)**2+(k*L)**2); out += np.exp(-d/xi)/d
    return out
arms = {}
for fn in sorted(glob.glob("lin_*.json")):
    d = json.load(open(fn)); arms.setdefault((d["s0"], d["L"], d["mu"]), []).append(d)
S = {}
for (s0, L, mu), ds in sorted(arms.items()):
    rows = []; fits = {}
    for d in ds:
        c = np.array([[x[0], x[1], x[2]] for x in d["cmp"]]); r, dat, lin = c.T
        m = (r >= 2) & (r <= L/2-2)
        big = m & (np.abs(lin) >= 0.1*np.abs(lin[r == 2][0]))      # поза околом нуля (post hoc)
        rel = (dat-lin)/lin
        k = np.sum(dat[m]*lin[m])/np.sum(lin[m]**2)                  # одна спільна амплітуда
        shape = np.max(np.abs(dat[m]-k*lin[m]))/np.abs(lin[r == 2][0])
        rows.append(dict(seed=d["seed"], relres=d["relres"], maxrel_all=float(np.max(np.abs(rel[m]))), maxrel_big=float(np.max(np.abs(rel[big]))),
                         ratio=float(k), shape_dev=float(shape), maxabs_frac=float(np.max(np.abs(dat[m]-lin[m]))/np.abs(lin[r == 2][0]))))
    d = ds[0]; c = np.array([[x[0], x[2]] for x in d["cmp"]]); r, lin = c.T
    for RM in (2, 3, 4):
        mm = (r >= RM) & (r <= L/2)
        try:
            p, _ = curve_fit(lambda rr, A, xi, C: A*yuk(rr, L, xi)+C, r[mm], lin[mm], p0=[0.1*np.sign(mu), 2.5, 0], bounds=([-50, 0.3, -1], [50, 60, 1]), maxfev=20000); fits[RM] = float(p[1])
        except Exception: fits[RM] = None
    key = f"s{s0:g}_L{L}_mu{mu:g}"
    S[key] = dict(n=len(rows), ratio_med=float(np.median([q["ratio"] for q in rows])), ratio_rng=[float(min(q["ratio"] for q in rows)), float(max(q["ratio"] for q in rows))],
                  maxrel_all=float(max(q["maxrel_all"] for q in rows)), maxrel_big=float(max(q["maxrel_big"] for q in rows)),
                  shape_dev=float(max(q["shape_dev"] for q in rows)), maxabs_frac=float(max(q["maxabs_frac"] for q in rows)), relres_max=float(max(q["relres"] for q in rows)),
                  xi_fit_lin=fits)
    print(key, json.dumps(S[key], ensure_ascii=False))
json.dump(S, open("ana470g.json", "w"), indent=1, ensure_ascii=False)
