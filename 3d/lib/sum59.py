import json,glob
for f in sorted(glob.glob('lc_s03.5_n100_L96_s*_tau8_D1_g*.json')):
    w=json.load(open(f))['w2']
    print(f,[(x['t'],x['nuc'],x['nB'],x.get('ndef'),round(x['boil'],2) if 'boil' in x else None) for x in w if x['t']%100==0])
