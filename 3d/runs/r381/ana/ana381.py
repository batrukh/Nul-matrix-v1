"""(381) аналіз скану s0 при α0,2: кінцеві стани t300."""
import sys, glob, json, ast; sys.argv = ['x']; sys.path[:0] = ['/mnt/project-files/model/s52/r351', '/mnt/project-files/model/s52']
import numpy as np, rq351f as Q
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
s, f, mod, z, psi = Q.run(L=16, check=1); N = 4096; out = []
for d in sorted(glob.glob('/mnt/project-files/model/s52/r381/s*_a0.2_seed*')):
    fn = glob.glob(d + '/t3*_ell4.npz'); c = np.load(fn[0], allow_pickle=True)
    A = np.abs(c['z'])**2; p = c['psi']; W = (p[f.tail] > 0) & (p[f.head] > 0); M = ~W
    nw = p > 0; e = W
    g = coo_matrix((np.ones(e.sum()), (f.tail[e], f.head[e])), shape=(N, N))
    nc, lab = connected_components(g, directed=False); sz = np.bincount(lab[nw], minlength=nc); worlds = sorted([int(x) for x in sz if x >= 50], reverse=True)
    ts, big = [], []
    for l in open(d + '/log'):
        if l.startswith("{'A2'"):
            r = ast.literal_eval(l); ts.append(r['t']); big.append(r['big'])
    r = dict(run=d.split('/')[-1], s0=float(d.split('/')[-1].split('_')[0][1:]), s0eff=float(A.mean()), AW=float(A[W].mean()) if W.any() else 0, AM=float(A[M].mean()), fW=float(W.mean()), fnode=float(nw.mean()), worlds=worlds, dead=float((A < .01).mean()), t=ts, big=big)
    r['lever'] = float((r['s0eff'] - r['AM'])/(r['AW'] - r['AM'])) if r['AW'] > r['AM'] else None
    out.append(r); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ('t', 'big')})
json.dump(out, open('/mnt/project-files/model/s52/r381/ana/ana381.json', 'w'))
