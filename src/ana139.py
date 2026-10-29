import ast, json, glob, numpy as np
def load(fn):
    s = open(fn).read(); i = s.index('{'); return ast.literal_eval(s[i:s.rindex('}')+1])
R = []
for d in sorted(glob.glob("L96_*")):
    x = load(d + "/log"); w = x["w2"]; _, l, g, s = d.split("_")
    R.append(dict(run=d, ell=float(l[1:]), g=float(g[1:]), seed=int(s[1:]), tr=[(q["t"], q["nuc"], q["big"]/96**2, q.get("Aout"), q["nB"]/96**2) for q in w], final=x["final"][:20]))
for s in (3, 4, 5):
    w = json.load(open(f"../lc_s03.5_n100_L96_s{s}_sk4l0.json"))
    R.append(dict(run=f"base_s{s}", ell=-1, g=0, seed=s, tr=[(q["t"], q["nuc"], q["big"]/96**2, q.get("Aout"), q["nB"]/96**2) for q in w["w2"]], final=w.get("final", [])[:20]))
json.dump(R, open("ana139.json", "w"))
for r in R:
    tr = r["tr"]; pick = [q for q in tr if int(q[0]) in (50, 100, 150, 200, 250)]
    print(r["run"], [(int(q[0]), q[1], round(q[2], 3), q[3]) for q in pick], "final", [v for v in r["final"] if v >= 20][:10], "frac", round(tr[-1][4], 3))
