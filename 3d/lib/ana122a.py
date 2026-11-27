"""ana122a: народження Z₂-вихорів (вакуум-свідомий детектор) у прогонах (120): частота на площу, глибина місця, зв'язок із ростом площі світу."""
import json, glob, re, numpy as np, matter98 as M
L = 96
def dd(p, q): return float(M.wrapd(np.array(p[0]), np.array(p[1]), q[0], q[1], L)[0])
rows = []
for f in sorted(glob.glob("*_gk1_gd_g104.json")):
    if "gv100" in f: continue
    m = re.search(r"_s(\d)_.*?(?:_b([\d.]+))?_pf", f); s = int(m.group(1)); b = float(m.group(2)) if m.group(2) else 1.5
    tr = json.load(open(f))["track"]; nb = 0; area = 0.0; bd = []; dH = np.zeros(31); per = []; nd = 0; nsum = 0.0
    for a, c in zip(tr, tr[1:]):
        if c["t"] < 30: continue
        k = sum(1 for p in c["cl"] if all(dd(p, q) > 2.5 for q in a["cl"])); nb += k; area += c["wtri"]
        nd += sum(1 for p in a["cl"] if all(dd(p, q) > 2.5 for q in c["cl"])); nsum += len(a["cl"])
        bd += [d for p, d in zip(c["cl"], c["dep"]) if all(dd(p, q) > 2.5 for q in a["cl"])]
        dH += np.array(c["dhist"]); per.append((c["wtri"] - a["wtri"], k, c["wtri"]))
    per = np.array(per, float); g = per[:, 0]
    # народження в одиниці, коли світ росте (Δплощі > 0) проти, коли ні — на 1000 трикутників
    up = g > 0; rate = lambda m_: 1000*per[m_, 1].sum()/per[m_, 2].sum() if m_.any() else np.nan
    Hn = dH/dH.sum(); bd = np.array(bd)
    rows.append(dict(beta=b, seed=s, births=nb, rate=1000*nb/area, life=nsum/max(nd, 1), area=area/len(per),
                     grow_rate=rate(up), still_rate=rate(~up), corr=float(np.corrcoef(g, per[:, 1])[0, 1]),
                     bdep_med=float(np.median(bd)) if len(bd) else None, bedge=float(np.mean(bd < 2)) if len(bd) else None, area_edge=float(Hn[:2].sum())))
rows.sort(key=lambda r: (r["beta"], r["seed"]))
for r in rows: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
json.dump(rows, open("ana122a.json", "w"))
