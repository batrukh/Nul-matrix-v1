import json,glob,numpy as np
out=[]
for f in sorted(glob.glob('runs/r*_s*.json')):
    d=json.load(open(f)); H=d['hist']; fl=[h['nlat']/4096 for h in H]
    dead=next((h['t'] for h in H if h['nW']>0 and h['rhoM']<0.5),None)
    f_at_dead=next((h['nlat']/4096 for h in H if h['t']==dead),None) if dead else None
    tstop=next((H[i]['t'] for i in range(len(H)) if H[i]['nlat']==H[-1]['nlat']),None)
    t95=next((h['t'] for h in H if h['nlat']/4096>=0.95),None); tW=next((h['t'] for h in H if h['nW']>0),None)
    out.append(dict(r=d['r'],seed=d['seed'],flat_end=round(fl[-1],3),t95=t95,tstop=tstop,t_world=tW,t_matrix_dead=dead,f_at_dead=f_at_dead,fW_end=round(H[-1]['nW']/4096,3),tP_end=H[-1]['tP']))
for o in out: print(o)
json.dump(out,open('ana458.json','w'),indent=1)
