import json,glob,numpy as np
for f in sorted(glob.glob('lc_*_L96_s*.json')):
    d=json.load(open(f)); w=d['w2']
    n=np.array([q['nB'] for q in w]); t=np.array([q['t'] for q in w]); h=t>150
    t1=next((q['t'] for q in w if q['nuc']>0),None)
    ain=[q['Ain'] for q in w if q['Ain'] and q['t']>150]
    bo=np.mean([q.get('boil',0) for q in w if q['t']>150])
    nuc=np.mean([q['nuc'] for q in w if q['t']>150])
    print(f,'t1',t1,'nB',round(n[h].mean()),'sd',round(n[h].std()),'max',n.max(),'nuc',round(nuc,1),'Ain',round(np.mean(ain),2) if ain else None,'boil',round(bo,2),'S',w[-1]['S'],'final',d['final'][:6])
    print('   ',[(q['t'],q['nB']) for q in w[::30]])
