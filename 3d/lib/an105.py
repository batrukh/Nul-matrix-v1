import json, glob, numpy as np, matter98 as M
L=96
def dist(a,b): return float(M.wrapd(np.array(a[0]),np.array(a[1]),b[0],b[1],L)[0])
out={}
for f in sorted(glob.glob("lc_s03.5_n100_L96_s*_tau8_D1_g0.02_iso*pf1_gz0.02*_g104.json")):
    if "gv100" in f and "b1.2" not in f: continue
    tr=json.load(open(f))["track"]; b=1.2 if "b1.2" in f else 1.3 if "b1.3" in f else 1.4 if "b1.4" in f else 1.5
    s=int(f.split("_L96_s")[1][0]); late=[r for r in tr if r["t"]>=150]
    rec=dict(beta=b, seed=s, n=float(np.mean([r["ncl"] for r in late])), wtri=float(np.mean([r["wtri"] for r in late])),
             dmax=float(np.mean([r["dmax"] for r in late])) if "dmax" in late[0] else None, t=[r["t"] for r in tr], ncl=[r["ncl"] for r in tr])
    if "dep" in tr[0]:
        bd=[]; alld=[]; H=np.zeros(31)
        for a,c in zip(tr,tr[1:]):
            if c["t"]<30: continue
            H+=np.array(c["dhist"])
            for p,d in zip(c["cl"],c["dep"]):
                alld.append(d)
                if all(dist(p,q)>2.5 for q in a["cl"]): bd.append(d)
        Hn=H/H.sum(); expect=float((Hn[:3]).sum())   # частка площі світу на глибині <3
        rec.update(nb=len(bd), birth_edge=float(np.mean(np.array(bd)<3)) if bd else None, area_edge=expect,
                   all_edge=float(np.mean(np.array(alld)<3)) if alld else None, birth_dep=float(np.median(bd)) if bd else None, world_dep_med=float(np.argmax(np.cumsum(Hn)>=0.5)))
    out[f"{b}_{s}"]=rec
    print({k:(round(v,2) if isinstance(v,float) else v) for k,v in rec.items() if k not in ("t","ncl")})
json.dump(out,open("an105.json","w"))
