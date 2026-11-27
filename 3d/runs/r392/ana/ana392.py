"""(392)/(393) аналіз: світ (≥50 вузлів) на t300, час народження (392), мінімум і кінцевий розмір світу (393); ряд big(t). Опора: (373)/(374) α0,1 з нуля."""
import json, ast, numpy as np
R='/mnt/project-files/model/s52/'
def log(d):
    L=[]
    for l in open(d+'/log'):
        if l.startswith('{'):
            try: L.append(ast.literal_eval(l.strip()))
            except Exception: pass
    return [x['t'] for x in L],[x['big'] for x in L]
out={}
for kind in ('late','off'):
    rr=[]
    for k in range(1,9):
        t,b=log(R+f'r392/runs/{kind}_seed{k}')
        rr.append(dict(seed=k,tend=t[-1],big=b[-1],frac=b[-1]/4096,tborn=next((ti for ti,bi in zip(t,b) if bi>=50),None),bmin=min(b),t=t,b=b))
    out[kind]=rr
    print(kind,[(r['seed'],r['tborn'],round(r['frac'],3),round(r['bmin']/4096,3)) for r in rr])
ref=[]
for k in range(1,9):
    t,b=log(R+f'r373/a0.1_seed{k}'); ref.append(dict(seed=k,t=t,b=b,tborn=next((ti for ti,bi in zip(t,b) if bi>=50),None)))
out['ref373']=ref
late=sum(r['big']>=50 for r in out['late']); off80=sum(r['frac']>=0.8 for r in out['off']); off50=sum(r['frac']<0.5 for r in out['off'])
print('(392) світ на t300:',late,'/8  ->', 'І-2 СПРОСТОВАНО' if late>=6 else ('І-2 ПІДТВЕРДЖЕНО' if late<=2 else 'нечітко'))
print('(393) світ ≥80 %:',off80,'/8; <50 % або зник:',off50,'/8 ->', '«α лише запал» ПІДТВЕРДЖЕНО' if off80>=6 else ('СПРОСТОВАНО' if off50>=6 else 'нечітко'))
print('час народження late:',[r['tborn'] for r in out['late']],' з нуля (373):',[r['tborn'] for r in ref])
json.dump(out,open(R+'r392/ana/ana392.json','w'))
