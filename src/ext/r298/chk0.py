"""(298) етап 0: (1) дійсний Φ, a≡0 → сили = gauge91 (b=0); (2) повна енергія зберігається ∝ dt² для комплексного випадкового стану; (3) калібрувальна інваріантність U(1)."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
from cgauge import CGauge, sym0, cayley
from gauge91 import Gauge
L = 8; r = np.random.default_rng(1)
wl = np.ones((3, L, L)); wt = np.ones((2, L, L))
P = sym0(r.standard_normal((L, L, 5, 5)))*0.3
U = np.array([cayley(0.5*(lambda X: X - np.swapaxes(X, -1, -2))(r.standard_normal((L, L, 5, 5)))) for _ in range(3)])
g0 = Gauge(np.full((L, L), -1.0), wl, wt, ka=1, kb=2, b=0.0); g1 = CGauge(np.full((L, L), -1.0), wl, wt)
F0P, F0E = g0.forces(P, U); F1P, F1E, F1A = g1.forces(P.astype(complex), U, np.zeros((3, L, L)))
print('control forces: dP', np.abs(F0P - F1P).max(), 'dE', np.abs(F0E - F1E).max(), 'FA', np.abs(F1A).max())
e0 = g0.edens(P, 0*P, U, 0*F0E).sum(); e1 = g1.edens(P.astype(complex), 0*P, U, 0*F0E, np.zeros((3, L, L)), np.zeros((3, L, L))).sum(); print('control energy', e0, e1)
# комплексний стан
Pc = sym0(r.standard_normal((L, L, 5, 5)) + 1j*r.standard_normal((L, L, 5, 5)))*0.3; A = r.standard_normal((3, L, L))*0.5
for dt in [0.02, 0.01, 0.005]:
    st = [Pc, 0*Pc, U, 0*F1E, A, 0*A]; Es = []
    for i in range(int(2/dt)):
        st = list(g1.step(*st, dt)); Es.append(g1.edens(*st).sum())
    Es = np.array(Es); print(f'dt={dt}: dE max {np.abs(Es - Es[0]).max():.3e}  E={Es[0]:.4f}')
# калібрувальна інваріантність U(1): Φ→e^{iθ}Φ, a_m(n) → a_m(n) + (θ_n − θ_{n+d})/q
from gauge91 import sh, DIRS
th = r.standard_normal((L, L)); A2 = np.array([A[m] + (th - sh(th, d)) for m, d in enumerate(DIRS)])
Pg = np.exp(1j*th)[..., None, None]*Pc
print('gauge inv energy', abs(g1.edens(Pc, 0*Pc, U, 0*F1E, A, 0*A).sum() - g1.edens(Pg, 0*Pc, U, 0*F1E, A2, 0*A).sum()))
