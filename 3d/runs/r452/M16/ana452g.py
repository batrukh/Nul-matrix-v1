"""(452Г) аналіз за criterion452g.txt"""
import json,glob,numpy as np
D="/mnt/project-files/model/s52/r452"
rows={};bad=0
for f in glob.glob(D+"/M16/lock_a*_L16.json"):
    d=json.load(open(f)); ref=json.load(open(f.replace("/M16/","/L16/")))
    # контроль: загальна f2 = зважене середнє класів
    for t0 in (10,20):
        n=sum(d[f"n_{c}2_{t0}"] for c in "MWF"); tot=sum((d[f"lock_{c}2_{t0}"] or 0)*d[f"n_{c}2_{t0}"] for c in "MWF")/n
        if abs(tot-ref[f"lock_s2_{t0}"])>1e-9: bad+=1
    rows.setdefault(d['ac'],[]).append(d)
def avg(v,c,minn=200):
    x=[d[f"lock_{c}2_{t0}"] for d in v for t0 in (10,20) if d[f"n_{c}2_{t0}"]>=minn and d[f"lock_{c}2_{t0}"] is not None]
    return (np.mean(x),np.std(x,ddof=1)/np.sqrt(len(x)) if len(x)>1 else 0,len(x)) if x else (None,None,0)
out={}
print("control mismatches:",bad)
print("α     f2M(±se,nwin)        f2F            f2W            nM2/nF2/nW2 (t20, mean)")
for a in sorted(rows):
    v=rows[a]; o={c:avg(v,c) for c in "MFW"}
    cnt=[np.mean([d[f"n_{c}2_20"] for d in v]) for c in "MFW"]
    out[a]=dict(M=o["M"],F=o["F"],W=o["W"],cnt=cnt)
    fm=lambda t:"—" if t[0] is None else f"{t[0]:.3f}±{t[1]:.3f}({t[2]})"
    print(f"{a:<5} {fm(o['M']):<20} {fm(o['F']):<14} {fm(o['W']):<14} {cnt[0]:.0f}/{cnt[1]:.0f}/{cnt[2]:.0f}")
g=lambda a,c:out[a][c][0]
A_=all(g(a,"M") is not None and g(a,"M")<=0.96 for a in (0.1,0.15))
B_=all(g(a,"M") is not None and g(a,"M")>=0.98 and (g(a,"F") is not None and g(a,"F")<=0.9) for a in (0.1,0.15))
ver="НЕДІЙСНО (контроль)" if bad else ("ПІДТВЕРДЖЕНО А" if A_ else ("ПІДТВЕРДЖЕНО Б" if B_ else "НЕЧІТКО"))
side=[(a,g(a,"W"),g(a,"M")) for a in (0.1,0.15,0.2)]
print("verdict:",ver,"| W vs M:",side)
json.dump(dict(rows={str(k):v for k,v in out.items()},verdict=ver,control_bad=bad,side=side),open(D+"/M16/ana452g.json","w"),indent=1,default=lambda x:None if x is None else float(x))
