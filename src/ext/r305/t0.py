import numpy as np, sys; sys.path.insert(0, '.'); import hc, plant
from scipy.sparse.linalg import eigsh, splu, LinearOperator
L = 48; c = np.array([0.5*L + 0.25*L, 0.5*np.sqrt(3)/2*L])
for R in [6, 8]:
    v1 = plant.snap(L, c + [R, 0]); v2 = plant.snap(L, c - [R, 0]); P, U = plant.plant(L, v1, v2)
    H = hc.build(P, U); lu = splu(H.tocsc()); Op = LinearOperator(H.shape, matvec=lu.solve, dtype=float)
    w, v = eigsh(Op, k=12, which='LM', ncv=60, tol=1e-10); w = np.sort(1/w)
    print(R, v1, v2, np.round(w, 5))
