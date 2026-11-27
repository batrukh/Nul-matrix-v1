import json,glob,numpy as np
fs=sorted(glob.glob('lc_s03.5_n100_L96_s[12].json'))+sorted(glob.glob('lc_s03.5_n100_L96_s*_tau*.json'))
for f in fs:
    d=json.load(open(f)); w=d['w2']; q=[x for x in w if 120<x['t']<=200]; N=96*96
    nuc=np.mean([x['nuc'] for x in q]); nB=np.mean([x['nB'] for x in q])
    print(f,'nuc',round(nuc,1),'per1000',round(nuc/N*1000,2),'frac',round(nB/N,3),'big',round(np.mean([x['big'] for x in q])),'boil',round(np.mean([x['boil'] for x in q if 'boil' in x]),2))
