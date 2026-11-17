"""(335) аналіз: база 8.11, розгортка запасу s0 (L96) і L192 (s0 3,5). Контроль s0 3,5 L96 = r276 g0.0025_s1–4 (T300).
Для кожного прогону: перше народження (nB>0), ряди nB/big/Aout/boil/hol у t 100/200/300, кінцеві світи з ψ (кількість, частка, найбільший, глибина)."""
import json, glob, numpy as np
from scipy import ndimage
def worlds(psi):
    L = int(round(np.sqrt(psi.size))); w = psi.reshape(L, L) > 0
    W = np.tile(w, (3, 3)); lab, _ = ndimage.label(W, structure=np.ones((3, 3)))
    c = lab[L:2*L, L:2*L]; ids = [i for i in np.unique(c) if i]; sizes = sorted([int((c == i).sum()) for i in ids], reverse=True)
    sizes = [s for s in sizes if s >= 20]
    dep = ndimage.distance_transform_edt(W)[L:2*L, L:2*L]
    return L, float(w.mean()), len(sizes), (sizes[0] if sizes else 0), float(dep.max())
runs = [(d, d.split('/')[-1]) for d in sorted(glob.glob('/mnt/project-files/model/s52/r335/s*_L*_s*'))]
runs += [(d, 'ref_s3.5_L96_' + d.split('_')[-1]) for d in sorted(glob.glob('/mnt/project-files/model/s52/r276/g0.0025_s[1-4]'))]
out = {}
for d, k in runs:
    f = glob.glob(d + '/*_le8.json')
    if not f: continue
    R = json.load(open(f[0])); w2 = R['w2']
    npz = [g for g in glob.glob(d + '/lc_*.npz') if not g.endswith(('_g104.npz', '_g91.npz', '_sn.npz', '_snap.npz'))][0]
    L, frac, nw, mx, dep = worlds(np.load(npz)['psi'])
    first = next((q['t'] for q in w2 if q.get('nB', 0) > 0), None)
    at = lambda t, key: (lambda q: q.get(key) if q else None)(next((q for q in w2 if abs(q['t'] - t) < 0.51), None))
    mean = lambda key, a, b: float(np.mean([q[key] for q in w2 if a <= q['t'] <= b and q.get(key) is not None])) if any(a <= q['t'] <= b and q.get(key) is not None for q in w2) else None
    o = dict(L=L, first=first, nB=[at(t, 'nB') for t in (100, 200, 300)], big=[at(t, 'big') for t in (100, 200, 300)],
             Aout=mean('Aout', 250, 300), A2=mean('A2', 250, 300), boil=mean('boil', 250, 300), hol=mean('hol', 250, 300), sd=mean('sd', 250, 300),
             boil_early=mean('boil', 10, 40), frac=round(frac, 3), nworld=nw, maxworld=mx, per1000=round(nw/(L*L)*1000, 2), depth=round(dep, 1))
    out[k] = o; print(k, o)
json.dump(out, open('/mnt/project-files/model/s52/r335/ana335.json', 'w'), indent=1)
