import numpy as np, sys, json
sys.path.insert(0, '.'); sys.path.insert(0, '/mnt/project-files/model/s52/r268/code')
import hc, phi90 as PH
from scipy.linalg import expm
out = {}
L = 24; I = np.broadcast_to(np.eye(5), (3, L, L, 5, 5)).copy()
P0 = np.diag(np.sort(np.linalg.eigvalsh(PH.vacuum(-1.0, 0.1)))); P = np.broadcast_to(P0, (L, L, 5, 5)).copy()
e = np.linalg.eigvalsh(hc.build(P, I, g=0.0).toarray()); out['g0_nzero'] = int((abs(e) < 1e-8).sum()); out['g0_max'] = float(e.max())
e = np.linalg.eigvalsh(hc.build(P, I, g=1.0).toarray()); out['g1_gap'] = float(abs(e).min())
rng = np.random.default_rng(1)
A = rng.standard_normal((L, L, 5, 5)); O = expm(A - np.swapaxes(A, -1, -2))
Pr = rng.standard_normal((L, L, 5, 5)); Pr = PH.sym0(Pr); Ur = expm(0.3*np.array([B - np.swapaxes(B, -1, -2) for B in rng.standard_normal((3, L, L, 5, 5))]))
e1 = np.linalg.eigvalsh(hc.build(Pr, Ur).toarray())
T = lambda X: np.swapaxes(X, -1, -2)
P2 = O @ Pr @ T(O); U2 = np.array([O @ Ur[m] @ T(hc.sh(O, d)) for m, d in enumerate([(1, 0), (0, 1), (-1, 1)])])
e2 = np.linalg.eigvalsh(hc.build(P2, U2).toarray()); out['gauge_inv_maxdiff'] = float(abs(e1 - e2).max())
print(json.dumps(out))
