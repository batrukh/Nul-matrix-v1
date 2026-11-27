import json,glob,numpy as np
for f in sorted(glob.glob('lc_*_L144_s*.json'))+sorted(glob.glob('lc_*_L96_s*.json')):
    if '_T1000' in f: continue
    d=json.load(open(f)); w=d['w2']; q=[x for x in w if 120<x['t']<=200]
    print(f,'nuc',round(np.mean([x['nuc'] for x in q]),1),'nB',round(np.mean([x['nB'] for x in q])),'big',round(np.mean([x['big'] for x in q])),'boil',round(np.mean([x.get('boil',np.nan) for x in q if 'boil' in x]),2),'final',d['final'][:15])
