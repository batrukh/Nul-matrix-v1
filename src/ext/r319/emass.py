"""(319) енергія вихорів r294 (пара на L96) без директора і з директором (fin_* r312) відносно відповідного вакууму."""
import numpy as np, sys, json
sys.path.insert(0, '/mnt/project-files/model/s52/r319'); from gd import GD
sys.path.insert(0, '/mnt/project-files/model/s52/r268/code'); import phi90 as PH
L = 96; z = np.load('/mnt/project-files/model/s52/r294/e0_ka1_kb2_a-1_L96.npz'); print('cx cy vy2', z['cx'], z['cy'], z['vy2'])
def E(P, U, n=None, mu=0.0, k=0.09, Lx=L):
    g = GD(-np.ones((Lx, Lx)), np.ones((3, Lx, Lx)), np.ones((2, Lx, Lx)), 1.0, 2.0, 0.1, mu=mu, k=k)
    e = g.edens(P, np.zeros_like(P), U, np.zeros_like(U))
    if n is not None: e = e + g.en(P, U, n)
    return e
Pv = PH.vacuum(-1.0, 0.1); ev0 = E(np.broadcast_to(Pv, (4, 4, 5, 5)).copy(), np.broadcast_to(np.eye(5), (3, 4, 4, 5, 5)).copy(), Lx=4)[0, 0]
eb = E(z['P'], z['U']); res = dict(base=float(np.sum(eb - ev0)))
for tag, mu, vt in [('mu001', 0.01, 'vac0.01'), ('mu001e2', 0.01, 'vac0.01'), ('mu003', 0.03, 'vac0.03'), ('mu003e2', 0.03, 'vac0.03')]:
    f = np.load(f'/mnt/project-files/model/s52/r312/fin_{tag}.npz'); v = np.load(f'/mnt/project-files/model/s52/r312/fin_{vt}.npz')
    evv = E(v['P'], v['U'], v['n'], mu=mu, Lx=v['P'].shape[0]).mean()
    res[tag] = float(np.sum(E(f['P'], f['U'], f['n'], mu=mu) - evv))
print(json.dumps(res, indent=1)); json.dump(res, open('emass.json', 'w'))
for t in res: print(t, 'відношення до бази', round(res[t]/res['base'], 4))
