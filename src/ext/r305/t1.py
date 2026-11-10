import numpy as np, sys, json; sys.path.insert(0, '.'); import hc, plant
from scipy.sparse.linalg import eigsh, splu, LinearOperator
L = 48; c = np.array([0.5*L + 0.25*L, 0.5*np.sqrt(3)/2*L]); R = 8; out = {}
v1 = plant.snap(L, c + [R, 0]); v2 = plant.snap(L, c - [R, 0])
for r0 in [0, 1, 2, 3, 4, 6, 1e9]:
    P, U = plant.thick(L, v1, v2, r0)
    H = hc.build(P, U); lu = splu(H.tocsc()); Op = LinearOperator(H.shape, matvec=lu.solve, dtype=float)
    w = np.sort(1/eigsh(Op, k=8, which='LM', ncv=48, tol=1e-10, return_eigenvectors=False))
    out[str(r0)] = w.tolist(); print(r0, np.round(w, 4), flush=True)
json.dump(out, open('t1.json', 'w'))
