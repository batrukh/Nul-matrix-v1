"""(302) довгий зонд зарядженої Аліси на L96: поштовх непарних Φ★ (гаус r0 3) у ядрі v1, запис Φ у r<12 кожні 1,0, шар r>30 від v1 (Π, E, EA).
Контрольні точки кожні 2000 кроків у ck/ (resume автоматично). python3 probe302.py EPS T"""
import sys, os, time, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H + '/../r298'); sys.path.insert(0, H + '/..')
from cgauge import CGauge, sym0
from gauge91 import cayley
from vac import G as VG
EPS, T = float(sys.argv[1]), float(sys.argv[2]); L = 96; dt = 0.05; TAG = 'kick' if EPS > 0 else 'bg'
i, j = np.meshgrid(np.arange(L), np.arange(L), indexing='ij'); X = i + 0.5*j; Y = (np.sqrt(3)/2)*j
def rel(x0, y0):
    best = None
    for p in (-1, 0, 1):
        for qq in (-1, 0, 1):
            dx, dy = X - x0 + p*L + qq*L/2, Y - y0 + qq*np.sqrt(3)/2*L; r = np.hypot(dx, dy)
            if best is None: best = [r, dx, dy]
            else: k = r < best[0]; best = [np.where(k, r, best[0]), np.where(k, dx, best[1]), np.where(k, dy, best[2])]
    return best
g = CGauge(np.full((L, L), -1.0), np.ones((3, L, L)), np.ones((2, L, L)), mu=0.3)
P0 = np.diag([-1, -1, -1, 1.5, 1.5]).astype(complex)
for _ in range(60000): P0 = P0 - 0.02*VG(P0, mu=0.3)
z = np.load(H + '/st302.npz'); v1 = tuple(z['v1'])
Hu, Hd = g.hol(z['U']); w0 = np.real((5 - np.trace(Hu, axis1=-2, axis2=-1)) + (5 - np.trace(Hd, axis1=-2, axis2=-1)))
r, dx, dy = rel(*v1); m = r < 6; s = (w0*m).sum(); cx, cy = v1[0] + (w0*m*dx).sum()/s, v1[1] + (w0*m*dy).sum()/s
r, dx, dy = rel(cx, cy); M = r < 12
gam = 0.3*np.clip((r - 30)/8, 0, 1)**2; gP = gam[..., None, None]; gE = gam[None, ..., None, None]; gA = gam[None]
iu = np.triu_indices(5); ck = H + f'/ck/{TAG}.npz'
if os.path.exists(ck):
    c = np.load(ck); P, Pi, U, E, A, EA, t, n = c['P'], c['Pi'], c['U'], c['E'], c['A'], c['EA'], float(c['t']), int(c['n']); rec = list(c['rec']); print('resume', t, flush=True)
else:
    P, U, A = z['P'].copy(), z['U'].copy(), z['A'].copy()
    zs = np.array([1, 1, -1, -1, 1.0]); Zm = np.outer(zs, zs) < 0
    rng = np.random.default_rng(7); K = rng.standard_normal((L, L, 5, 5)) + 1j*rng.standard_normal((L, L, 5, 5))
    K = 0.5*(K + np.swapaxes(K, -1, -2))*Zm; P = sym0(P + EPS*K*np.exp(-r**2/18)[..., None, None])
    Pi = np.zeros_like(P); E = np.zeros_like(U); EA = np.zeros_like(A); t = 0.0; n = 0; rec = []
t0 = time.time()
while t < T - 1e-9:
    if n % 20 == 0: rec.append(((P - P0)[..., iu[0], iu[1]])[M].astype(np.complex64))
    FP, FE, FEA = g.forces(P, U, A)
    Pi = Pi + 0.5*dt*(FP - gP*Pi); E = E + 0.5*dt*(FE - gE*E); EA = EA + 0.5*dt*(FEA - gA*EA)
    P = sym0(P + dt*Pi); U = cayley(dt*E) @ U; A = A + dt*EA
    FP, FE, FEA = g.forces(P, U, A)
    Pi = (Pi + 0.5*dt*FP)/(1 + 0.5*dt*gP); E = (E + 0.5*dt*FE)/(1 + 0.5*dt*gE); EA = (EA + 0.5*dt*FEA)/(1 + 0.5*dt*gA)
    t += dt; n += 1
    if n % 2000 == 0:
        np.savez(ck + '.tmp.npz', P=P, Pi=Pi, U=U, E=E, A=A, EA=EA, t=t, n=n, rec=np.array(rec)); os.replace(ck + '.tmp.npz', ck)
        print(round(t), round(time.time() - t0), flush=True)
np.savez(H + f'/probe302_{TAG}.npz', rec=np.array(rec), dx=dx[M], dy=dy[M], cx=cx, cy=cy, dts=dt*20)
U_cut = np.array([np.linalg.norm(U[m] - np.eye(5), axis=(-2, -1)) for m in range(3)]); np.save(H + f'/ucut_{TAG}.npy', U_cut)
print('done', TAG, round(cx, 2), round(cy, 2))
