import json,glob,numpy as np
for f in sorted(glob.glob('lc_s03.5_n100_L96_s[12]_tau8.json'))+sorted(glob.glob('lc_s03.5_n100_L96_s*_tau8_rnd*.json')):
    d=json.load(open(f)); w=d['w2']; q=[x for x in w if 120<x['t']<=200]; N=96*96
    nuc=np.mean([x['nuc'] for x in q]); nB=np.mean([x['nB'] for x in q])
    print(f,'nuc',round(nuc,1),'per1000',round(nuc/N*1000,2),'frac',round(nB/N,3),'boil',round(np.mean([x['boil'] for x in q if 'boil' in x]),2),'series',[(x['t'],x['nuc'],x['nB']) for x in w if x['t']%40==0])
