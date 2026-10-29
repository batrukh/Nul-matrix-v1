"""ana69: № 69 (розд. 4.4 (134)). Базова модель local60 без швів (sk=4), s0=3,5, n=100, T250; L = 48, 72, 96, 144.
Нові прогони — r69/; наявні контролі — ../s52/r129 (L96 сіди 1–2) і ../s52/r46 (L144 сіди 1–3).
Для кожного прогону: світи наприкінці (final, ≥ 20 вузлів), частка вузлів у світах, густина світів на 1000 вузлів,
кипіння і запас поза світами (Aout) наприкінці, хід nuc(t), big(t)/L²."""
import json, glob, sys, collections
import numpy as np
P = sys.argv[1] if len(sys.argv) > 1 else "."
fs = glob.glob("lc_*.json") + glob.glob(P + "/r129/lc_s03.5_n100_L96_s*_sk4l0.json") + glob.glob(P + "/r46/lc_s03.5_n100_L144_s*_sk4l96.json")
rows = []
for fn in sorted(fs):
    d = json.load(open(fn)); nm = d["name"]; L = int(nm.split("_L")[1].split("_")[0]); seed = int(nm.split("_s")[2].split("_")[0])
    N = L*L; w = [x for x in d["final"] if x >= 20]; last = d["w2"][-1]
    tr = [(q["t"], q.get("nuc", 0), q.get("big", 0)/N, q.get("nB", 0)/N, q.get("Aout"), q.get("boil")) for q in d["w2"]]
    rows.append(dict(L=L, seed=seed, nw=len(w), frac=sum(w)/N, dens=1000*len(w)/N, big=(max(w)/N if w else 0), sizes=w,
                     boil=last.get("boil"), Aout=last.get("Aout"), Ain=last.get("Ain"), tr=tr, src=(fn.split("/")[-2] if "/" in fn else "r69")))
by = collections.defaultdict(list)
for r in rows: by[r["L"]].append(r)
print("L  seed  світів  частка  густ/1000  найбільший  кипіння  Aout  Ain  розміри")
for L in sorted(by):
    for r in sorted(by[L], key=lambda r: r["seed"]):
        print(f"{L:3d} {r['seed']:2d} {r['nw']:4d} {r['frac']:6.2f} {r['dens']:7.2f} {r['big']:7.2f} {r['boil'] or 0:7.3f} {r['Aout'] or 0:5.2f} {r['Ain'] or 0:5.2f}  {r['sizes'][:8]}")
    a = lambda k: np.array([r[k] for r in by[L]], float)
    print(f"  L{L} середнє: світів {a('nw').mean():.1f}±{a('nw').std():.1f}, частка {a('frac').mean():.2f}±{a('frac').std():.2f}, густина {a('dens').mean():.2f}, найбільший {a('big').mean():.2f}")
json.dump(rows, open("r69/ana69.json", "w"))
