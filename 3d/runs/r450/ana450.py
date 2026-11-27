"""(450) вердикт + опис: частка вузлів-дефектів (CNA не-ОЦК) у світі проти ґратки; час народження."""
import sys, json, glob, numpy as np
sys.path.insert(0, '/mnt/project-files/model/s52/r421'); import classify as C
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
from scipy.spatial import Delaunay
out = []
for k in range(1, 9):
    z = np.load(f'/mnt/project-files/model/s52/r448/runs/R_e0.25_s{k}.npz'); P = z['P']; L = float(np.atleast_1d(z['box'])[0])
    _, lab = C.cna(P, L); dfc = np.asarray(lab) != 'bcc'
    f = [x for x in glob.glob(f'runs/s{k}/*.npz') if not x.endswith('_ck.npz')][0]
    psi = np.load(f)['psi']; d = json.load(open(f[:-4] + '.json')); w = d['w2']
    big = w[-1]['big']; n = len(psi); W = psi > 0
    r = dict(seed=k, n=n, world=round(big/n, 3), is_world=bool(big/n > 0.5), t_end=w[-1]['t'],
             t_birth=next((q['t'] for q in w if q['big'] >= 50), None),
             defect_lat=round(float(dfc.mean()), 3), defect_in_world=round(float(dfc[W].mean()), 3),
             world_on_defects=round(float(W[dfc].mean()), 3), world_on_bcc=round(float(W[~dfc].mean()), 3))
    out.append(r); print(r, flush=True)
json.dump(out, open('ana450.json', 'w'), indent=1)
print('worlds', sum(r['is_world'] for r in out), '/8')
