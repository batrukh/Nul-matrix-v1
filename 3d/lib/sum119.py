import json, glob, sys
for fn in sorted(glob.glob('rq_L16_s*_n100_*.json')):
    if 'ac' in fn: continue
    w=[r for r in json.load(open(fn))['w2'] if 'flat' in r]; L=w[-20:]
    fin=sum(r['nB'] for r in L)/len(L)/4096*100
    first=next((r['t'] for r in w if r['nuc']>0), None)
    print(fn, 'T',w[-1]['t'],'maxB%',round(max(r['nB'] for r in w)/40.96,1),'fin%',round(fin,1),'nuc_end',w[-1]['nuc'],'top',w[-1]['top'][:4],'first',first,'flat',round(sum(r['flat'] for r in L)/len(L),3),'boil',round(sum(r.get('boil',0) for r in L)/len(L),3))
