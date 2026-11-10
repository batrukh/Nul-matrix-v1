import numpy as np, sys; sys.path.insert(0, '.'); from gd import *
from scipy.linalg import expm
L = 8; rng = np.random.default_rng(0)
g = GD(-np.ones((L, L)), np.ones((3, L, L)), np.ones((2, L, L)), 1.0, 2.0, 0.1, mu=0.7, k=0.3)
P = sym0(rng.normal(size=(L, L, 5, 5))*0.3); P = 0.5*(P + T_(P))
U = np.array([expm(PA(rng.normal(size=(5, 5)))*0.5) for _ in range(3*L*L)]).reshape(3, L, L, 5, 5)
n = rng.normal(size=(L, L, 5)); n /= np.linalg.norm(n, axis=-1, keepdims=True)
E0 = lambda P, U, n: np.sum(g.en(P, U, n))
FP, FE, Fn = g.forces_d(P, U, n); FP0, FE0 = g.forces(P, U)
eps = 1e-6; i, j = 3, 4
dP = np.zeros_like(P); M = np.zeros((5, 5)); M[1, 2] = M[2, 1] = 1; dP[i, j] = sym0(M)
num = -(E0(P + eps*dP, U, n) - E0(P - eps*dP, U, n))/(2*eps); print('P', num, np.sum((FP - FP0)[i, j]*dP[i, j]))
dn = np.zeros_like(n); v = rng.normal(size=5); v -= v@n[i, j]*n[i, j]; dn[i, j] = v
num = -(E0(P, U, n + eps*dn) - E0(P, U, n - eps*dn))/(2*eps); print('n', num, np.sum(Fn[i, j]*v))
m = 1; A = PA(rng.normal(size=(5, 5))); Up = U.copy(); Um = U.copy(); Up[m, i, j] = expm(eps*A) @ U[m, i, j]; Um[m, i, j] = expm(-eps*A) @ U[m, i, j]
num = -(E0(P, Up, n) - E0(P, Um, n))/(2*eps); print('U', num, np.sum((FE - FE0)[m, i, j]*A)*(g.ka/2))
