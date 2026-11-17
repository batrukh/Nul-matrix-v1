"""(353) review scans (alpha, ell_tau, seeds 5-8); same analysis as ana348.
For every run dir (r346 with field, r348 without) read *_sp.npz (frames every 10 units: t, psi, z) and compute per frame:
N domains (psi>0 clusters >=5 nodes, triangular 6-neighbour adjacency, periodic), domain fraction, mean size, largest,
size list, and link intensities inside (both ends psi>0) / outside (both ends psi<0) -> A_in direct.
Usage: python3 ana348.py DIR [DIR ...]  -> writes ana353.json next to this script (merged)."""
import sys, os, glob, json, numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../r348/code'))
import simNlib as N
ST = np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]])          # triangular lattice in skew index coordinates
_fab = {}
def fab(L):
    if L not in _fab: _fab[L] = N.Fabric(L)
    return _fab[L]
def clusters(psi, L, mn=5):
    w = psi.reshape(L, L) > 0
    lab, n = ndimage.label(np.tile(w, (3, 3)), structure=ST)
    c = lab[L:2*L, L:2*L]; ids, cnt = np.unique(c[c > 0], return_counts=True)
    return sorted([int(x) for x in cnt if x >= mn], reverse=True), float(w.mean())
def frame(psi, z, L):
    f = fab(L); sz, frac = clusters(psi, L)
    A = np.abs(z)**2; pin = psi > 0; a, b = f.tail, f.head
    lin, lout = pin[a] & pin[b], (~pin[a]) & (~pin[b])
    return dict(N=len(sz), frac=round(frac, 4), mean=round(float(np.mean(sz)), 1) if sz else 0.0, big=sz[0] if sz else 0,
                sizes=sz, Ain=round(float(A[lin].mean()), 3) if lin.any() else None, Aout=round(float(A[lout].mean()), 3), Amean=round(float(A.mean()), 4))
out_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ana353.json')
out = json.load(open(out_p)) if os.path.exists(out_p) else {}
for d in sys.argv[1:]:
    sp = glob.glob(os.path.join(d, '*_sp.npz'))
    if not sp: print('no sp', d); continue
    D = np.load(sp[0]); L = int(round(np.sqrt(D['psi'].shape[1])))
    key = os.path.basename(os.path.dirname(os.path.abspath(d))) + '/' + os.path.basename(os.path.abspath(d))
    out[key] = dict(L=L, t=[float(x) for x in D['t']], fr=[frame(p, z, L) for p, z in zip(D['psi'], D['z'])])
    last = out[key]['fr'][-1]; print(key, 't', out[key]['t'][-1], {k: last[k] for k in ('N', 'frac', 'mean', 'big', 'Ain', 'Aout', 'Amean')})
json.dump(out, open(out_p, 'w'))
