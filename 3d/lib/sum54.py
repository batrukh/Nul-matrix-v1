import json,glob,numpy as np
for f in sorted(glob.glob('lc_*_L96_s*_T1000.json')):
    d=json.load(open(f)); w=json.load(open(f.replace('_T1000','')))['w2']+d['w2']
    for a,b in ((150,300),(300,500),(500,700),(700,900),(900,1001)):
        q=[x for x in w if a<x['t']<=b]
        print(f[:-5],a,b,'nB',round(np.mean([x['nB'] for x in q])),'nuc',round(np.mean([x['nuc'] for x in q]),1),'big',round(np.mean([x['big'] for x in q])),'boil',round(np.mean([x.get('boil',np.nan) for x in q if 'boil' in x]),2))
    print('  final',d['final'][:8],'S',d['w2'][-1]['S'])
