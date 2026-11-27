"""(479) аналіз: світ на t300 (>=50), час народження, розмір, стабільність t200->t300; Wilson."""
import ast,glob,json,re,numpy as np
from math import sqrt
R='/mnt/project-files/model/s52/r479/'
def wil(k,n,z=1.96):
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return round(c-h,2),round(c+h,2)
out={}
for d in sorted(glob.glob(R+'L*_seed*')):
    m=re.search(r'L(\d+)_a([\d.]+)_seed(\d+)',d); L,a,s=int(m[1]),m[2],int(m[3]); N=L**3
    D=[ast.literal_eval(l) for l in open(d+'/log') if l.startswith('{')]
    t=np.array([x['t'] for x in D]); b=np.array([x['big'] for x in D])
    if t[-1]<300: continue
    tb=next((x['t'] for x in D if x['big']>=50),None)
    b200=b[np.argmin(abs(t-200))]
    out.setdefault(f'L{L}_a{a}',[]).append(dict(seed=s,world=bool(b[-1]>=50),frac=round(b[-1]/N,4),t_birth=tb,ch200_300=None if b200==0 else round(b[-1]/b200-1,3),A2=round(D[-1]['A2'],4)))
res={}
for k,v in out.items():
    n=len(v);kk=sum(x['world'] for x in v)
    res[k]=dict(n=n,worlds=kk,wilson=wil(kk,n) if n else None,runs=v)
    print(k,f'{kk}/{n}',wil(kk,n))
    for x in v: print('  ',x)
json.dump(res,open(R+'ana479.json','w'),indent=1)
