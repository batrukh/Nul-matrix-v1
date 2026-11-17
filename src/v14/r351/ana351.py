"""(351) аналіз: 3D тетраедри Куна, база 8.11, прохід s0. Світ = ψ>0 зв'язна ділянка ≥50 вузлів, живе ≥20 од. (nB/big у w2)."""
import json, glob, numpy as np
out = {}
for fn in sorted(glob.glob('/mnt/project-files/model/s52/r351/runs/*/t3_*.json')):
    j = json.load(open(fn)); w = j['w2']; s0 = j['s0']; seed = fn.split('seed')[1].split('/')[0]
    psi = np.load(fn.replace('.json', '.npz'))['psi']
    late = [r for r in w if r['t'] >= 50]
    out[f'{s0}_{seed}'] = dict(s0=s0, seed=int(seed), t=w[-1]['t'], max_big=max(r['big'] for r in w), max_nB=max(r['nB'] for r in w),
        npos=int((psi > 0).sum()), psimax=float(psi.max()), psimean=float(psi.mean()),
        proto=float(np.mean([r['proto'] for r in late])), boil=float(np.nanmean([r.get('boil',np.nan) for r in late])),
        flat=float(np.mean([r['flat'] for r in late])), A2=float(np.mean([r['A2'] for r in late])),
        tser=[r['t'] for r in w], proto_t=[r.get('proto',np.nan) for r in w], boil_t=[r.get('boil',np.nan) for r in w])
    o = out[f'{s0}_{seed}']; print(s0, seed, o['t'], o['max_big'], o['npos'], round(o['psimax'], 3), round(o['proto'], 3), round(o['boil'], 3), round(o['flat'], 3), round(o['A2'], 3))
json.dump(out, open('/mnt/project-files/model/s52/r351/ana351.json', 'w'))
