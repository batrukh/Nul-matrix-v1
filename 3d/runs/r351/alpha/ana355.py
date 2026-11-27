"""(355) аналіз проходу обертання α у 3D (база 8.11, L16 T100, seed 1). Те саме, що ana351, + α=1 з (351) seed 1 як точка відліку."""
import json, glob, numpy as np
out = {}
fs = sorted(glob.glob('/mnt/project-files/model/s52/r351/alpha/s*_a*/t3_*.json'))
fs += [f for f in glob.glob('/mnt/project-files/model/s52/r351/runs/s*_seed1/t3_*.json') if ('s3_' in f or 's3.5_' in f)]
for fn in fs:
    j = json.load(open(fn)); w = j['w2']; s0 = j['s0']
    ac = float(fn.split('_a')[1].split('/')[0]) if '/alpha/' in fn else 1.0
    psi = np.load(fn.replace('.json', '.npz'))['psi']; late = [r for r in w if r['t'] >= 50]
    k = f'{s0}_{ac}'
    out[k] = dict(s0=s0, ac=ac, t=w[-1]['t'], max_big=max(r['big'] for r in w), max_nB=max(r['nB'] for r in w),
        npos=int((psi > 0).sum()), psimax=float(psi.max()), psimean=float(psi.mean()),
        proto=float(np.mean([r['proto'] for r in late])), protomax=float(max(r['proto'] for r in w)),
        boil=float(np.nanmean([r.get('boil', np.nan) for r in late])), flat=float(np.mean([r['flat'] for r in late])),
        hol=float(np.mean([r['hol'] for r in late])),
        tser=[r['t'] for r in w], flat_t=[r['flat'] for r in w], boil_t=[r.get('boil', np.nan) for r in w], proto_t=[r['proto'] for r in w])
    o = out[k]; print(s0, ac, o['t'], o['max_big'], o['npos'], round(o['psimax'], 3), round(o['protomax'], 3), round(o['boil'], 3), round(o['flat'], 3), round(o['hol'], 3))
json.dump(out, open('/mnt/project-files/model/s52/r351/alpha/ana355.json', 'w'))
