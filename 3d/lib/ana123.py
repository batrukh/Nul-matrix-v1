"""ana123: (123) кубічний член b потенціалу Φ★. Для b = 0 (прогони (120), T=250) і b = 0,1; 0,3 (run123): Кіббл (t ≤ 20),
стала кількість вихорів (t ≥ 150), народження на 1000 трикутників світу за одиницю (t ≥ 30), f3 — частка світу у вакуумі −Φ★,
кінцевий стан: вихорів усього / поза стінкою, частка світу на стінці."""
import json, glob, re, numpy as np, gf104 as GFm, matter98 as M, phi90 as PH, gauge91 as G
L = 96; X, Y = PH.coords(L); sh = G.sh
def dd(p, q): return float(M.wrapd(np.array(p[0]), np.array(p[1]), q[0], q[1], L)[0])
res = {}
for f in sorted(glob.glob("*_gk1_gd*_g104.json")):
    if "gv100" in f or "_sn_gk1_gd_g104" in f: continue
    m = re.search(r"_s(\d)_.*?(?:_b([\d.]+))?_pf.*?(?:_pb([\d.]+))?_g104", f); be = m.group(2) or "1.5"
    if be not in ("1.3", "1.5"): continue
    key = f"pb{m.group(3) or '0'}_b{be}_s{m.group(1)}"; tr = json.load(open(f))["track"]
    t = np.array([r["t"] for r in tr]); n = np.array([r["ncl"] for r in tr]); w = np.array([r["wtri"] for r in tr])
    f3 = np.array([r.get("f3") if r.get("f3") is not None else np.nan for r in tr], float)
    nb = 0; area = 0.0
    for a, c in zip(tr, tr[1:]):
        if c["t"] < 30: continue
        nb += sum(all(dd(p, q) > 2.5 for q in a["cl"]) for p in c["cl"]); area += c["wtri"]*(c["t"] - a["t"])
    d = np.load(f.replace(".json", ".npz")); GF = GFm.GF(L, 1); GF.P = d["P"]; GF.U = d["U"]
    det, wm = GF.defects(d["psi"], aware=True); v3 = (np.linalg.eigvalsh(d["P"]) > 0).sum(-1) >= 3
    mx = np.array([(v3 != sh(v3, (1, 0))) | (v3 != sh(v3, (0, 1))), (sh(v3, (1, 0)) != sh(v3, (1, 1))) | (sh(v3, (1, 1)) != sh(v3, (0, 1)))])
    res[key] = dict(kib=int(n[t <= 20].max()), late=float(n[t >= 150].mean()), birth=1000*nb/max(area, 1), f3_20=float(np.nanmean(f3[(t >= 15) & (t <= 25)])),
                    f3_late=float(np.nanmean(f3[t >= 150])), wtri=float(w[t >= 150].mean()),
                    end_all=len(M.clusters(np.where((det < 0) & wm, -1.0, 1.0), X, Y, L)), end_pure=len(M.clusters(np.where((det < 0) & wm & ~mx, -1.0, 1.0), X, Y, L)),
                    wall=float((mx & wm).sum()/max(wm.sum(), 1)), t=t.tolist(), n=n.tolist(), f3=[None if np.isnan(x) else x for x in f3])
    print(key, {k: round(v, 3) if isinstance(v, float) else v for k, v in res[key].items() if k not in ("t", "n", "f3")}, flush=True)
json.dump(res, open("ana123.json", "w"))
