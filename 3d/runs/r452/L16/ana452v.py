"""(452В) аналіз: f2(α) середнє ± с.п. по зернах, порівняння з L8, вердикт за criterion452v.txt"""
import json,glob,numpy as np,re
D="/mnt/project-files/model/s52/r452"
def load(pat):
    r={}
    for f in glob.glob(pat):
        d=json.load(open(f)); r.setdefault(d['ac'],[]).append(d)
    return r
L16=load(D+"/L16/lock_a*_s*_L16.json"); L8=load(D+"/lock_a*_s?.json")
out={}
print("α  n  f2(L16)±se  f0  twist2  psimax(med, n>0)  | f2(L8)")
for a in sorted(L16):
    v=L16[a]; f2=np.array([(d['lock_s2_10']+d['lock_s2_20'])/2 for d in v]); f0=np.array([(d['lock_s0_10']+d['lock_s0_20'])/2 for d in v])
    tw=np.array([d['twist_s2'] for d in v]); pm=np.array([d['psimax'] for d in v])
    l8=[(d['lock_s2_10']+d['lock_s2_20'])/2 for d in L8.get(a,[])]
    out[a]=dict(n=len(v),f2=f2.mean(),se=f2.std(ddof=1)/np.sqrt(len(v)) if len(v)>1 else 0,f0=f0.mean(),tw=tw.mean(),psimax=float(np.median(pm)),nworld=int((pm>0).sum()),f2L8=float(np.mean(l8)) if l8 else None)
    o=out[a]; print(f"{a:<5}{o['n']:>2} {o['f2']:.3f}±{o['se']:.3f} {o['f0']:.3f} {o['tw']:.4f} {o['psimax']:+.2f} {o['nworld']} | {o['f2L8']}")
A=sorted(out)
c1=all(out[a]['f2']>=0.98 for a in A if a<=0.15)
first=[a for a in A if out[a]['f2']<0.98]; first=first[0] if first else None
c2=first in (0.2,0.25)
c3=0.80<=out.get(0.4,{}).get('f2',-1)<=0.97
ref=(out.get(0.4,{}).get('f2',1)<0.7) or any(out[a]['f2']<0.95 for a in A if a<=0.15)
ver="ПІДТВЕРДЖЕНО" if (c1 and c2 and c3) else ("СПРОСТОВАНО" if ref else "НЕЧІТКО")
r=out.get(0.1,{}).get('tw',np.nan)/out.get(0.05,{}).get('tw',np.nan)
print("c1",c1,"first",first,"c2",c2,"c3",c3,"refute",ref,"->",ver,"| twist(0.1)/twist(0.05)=",round(r,2))
json.dump(dict(rows={str(k):v for k,v in out.items()},c1=c1,first=first,c2=c2,c3=c3,verdict=ver,twist_ratio=r),open(D+"/L16/ana452v.json","w"),indent=1,default=float)
