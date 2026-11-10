"""(302) стан L96: дві «заряджені Аліси» (Аліса + оберт ±1 в одній точці) на відстані L/2·√3/2 ≈ 41,6 по вертикалі; релаксація з тертям 0,2 до T."""
import numpy as np, sys, os, time
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H + '/../r298'); sys.path.insert(0, H + '/..')
from cgauge import CGauge, sym0
from gauge91 import DIRS, sh
from vac import G as VG
T = float(sys.argv[1]); L = 96; dt = 0.05; q = 1.0; RHO = 3.0
wrap = lambda x: (x + np.pi) % (2*np.pi) - np.pi
P0 = np.diag([-1, -1, -1, 1.5, 1.5]).astype(complex)
for _ in range(60000): P0 = P0 - 0.02*VG(P0, mu=0.3)
i, j = np.meshgrid(np.arange(L), np.arange(L), indexing='ij'); X = i + 0.5*j; Y = (np.sqrt(3)/2)*j
i0, j0 = 48 + 0, 24; i1, j1 = i0 - 24, j0 + 48          # v1, v2: v2 над v1, відстань 48·√3/2 = 41,6
v1 = (float(X[i0, j0]), float(Y[i0, j0])); v2 = (float(X[i1, j1]), float(Y[i1, j1]))
M = ((v1[0] + v2[0])/2, (v1[1] + v2[1])/2); mdx = X - M[0]; mdy = Y - M[1]          # середина пари в межах ґратки, без образів
th = np.zeros((L, L)); h = np.ones((L, L)); f = np.ones((L, L))
for (x0, y0), sg in ((v1, 1), (v2, -1)):
    dx, dy = X - x0, Y - y0; th += sg*np.arctan2(dy, dx); rr = np.hypot(dx, dy); h *= 1 - np.exp(-rr**2/RHO**2); f *= np.tanh(rr/2)
rM = np.hypot(mdx, mdy); th *= np.clip((40 - rM)/10, 0, 1)       # вікно: θ неперервна на торі (пара в межах r<21 від середини)
P = sym0((f*np.exp(1j*th))[..., None, None]*P0 + 1e-3*np.random.default_rng(1).standard_normal((L, L, 5, 5)))
A = np.array([-wrap(sh(th, d) - th)/q*0.5*(h + sh(h, d)) for d in DIRS])
U = np.broadcast_to(np.eye(5), (3, L, L, 5, 5)).copy(); Z = np.diag([1., 1., -1., -1., 1.]); xc_ = v1[0] + 0.25
for m, ev in enumerate([(1.0, 0.0), (0.5, np.sqrt(3)/2), (-0.5, np.sqrt(3)/2)]):
    s_ = (xc_ - X)/ev[0]; yc = Y + s_*ev[1]; hit = (s_ > 0) & (s_ < 1) & (yc > v1[1]) & (yc < v2[1]); U[m][hit] = Z
g = CGauge(np.full((L, L), -1.0), np.ones((3, L, L)), np.ones((2, L, L)), mu=0.3)
Pi = np.zeros_like(P); E = np.zeros_like(U); EA = np.zeros_like(A); t = 0.0; t0 = time.time()
while t < T - 1e-9:
    P, Pi, U, E, A, EA = g.step(P, Pi, U, E, A, EA, dt, gam=0.2); t += dt
    if abs(t % 25) < dt/2: print(f't={t:.0f} E={g.edens(P, Pi, U, E, A, EA).sum():.3f} ({time.time()-t0:.0f}s)', flush=True)
np.savez_compressed('st302.npz', P=P, U=U, A=A, v1=v1, v2=v2)
print('done')
