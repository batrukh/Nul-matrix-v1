import sys, time, numpy as np
sys.path.insert(0, '/mnt/project-files/model/s52/r268/code'); sys.path.insert(0, '.')
import gauge91 as G, phi90 as PH
L = 48; rng = np.random.default_rng(1)
P = PH.sym0(0.3*rng.standard_normal((L, L, 5, 5))); Pi = PH.sym0(0.1*rng.standard_normal((L, L, 5, 5)))
def rot(): 
    A = 0.3*rng.standard_normal((3, L, L, 5, 5)); A = A - np.swapaxes(A, -1, -2); return G.cayley(A)
U = rot(); E = 0.1*rng.standard_normal((3, L, L, 5, 5)); E = E - np.swapaxes(E, -1, -2)
psi = rng.standard_normal((L, L)); chi = 0.5*(1 + np.tanh(psi/0.2)); wl, wt = G.weights(chi, 0.02)
g = G.Gauge(1.0 - 2.0*chi, wl, wt, 1.0, 2.0, 0.1)
t = time.time(); r0 = g.step(P, Pi, U, E, 0.05, gam=0.2); r0 = g.step(*r0, 0.05, gam=0.2); t0 = time.time() - t
F0 = g.forces(P, U)
import gauge_nb; gauge_nb.install()
g.step(P, Pi, U, E, 0.05, gam=0.2)   # компіляція
t = time.time(); r1 = g.step(P, Pi, U, E, 0.05, gam=0.2); r1 = g.step(*r1, 0.05, gam=0.2); t1 = time.time() - t
F1 = g.forces(P, U)
print('forces diff', [float(np.abs(a - b).max()) for a, b in zip(F0, F1)])
print('2 steps diff', [float(np.abs(a - b).max()) for a, b in zip(r0, r1)], 'time', round(t0, 3), round(t1, 3), 'x', round(t0/t1, 1))
