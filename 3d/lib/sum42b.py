import json,glob
for f in sorted(glob.glob('lc_s03.5_n100_L96_s*_tau8*.json')):
    w=json.load(open(f))['w2']
    print(f,[(x['t'],x['nuc'],x['nB'],x.get('next'),x.get('nfl'),round(x['boil'],2) if 'boil' in x else None) for x in w if x['t']%25==0])
