"""(300) вихор Аліси + вихор з обертом фази (+1) у SO(5)×U(1) (cgauge з r298).
MODE: both (Аліса v1/v2 + оберт +1 на відстані D від v1 по x, −1 далеко), alice (лише Аліса), wind (лише ±1, ті самі місця).
Аліса — калібрувальна посадка як (285): Φ★ однорідне, ланки через вертикальний відрізок між v1 і v2 несуть Z = diag(1,1,−1,−1,1).
Міри: енергія; відстань від v1 до центру потоку B поблизу v1; число обертів n біля v1; |Φ|² у v1."""
import numpy as np, sys, os, json, time
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H + '/../r298'); sys.path.insert(0, H + '/..')
from cgauge import CGauge, sym0
from gauge91 import DIRS, sh
from vac import G as VG
MODE = sys.argv[1]; D = float(sys.argv[2]); T = float(sys.argv[3]); GAM = 0.2; TG = 100.0
L = 64; dt = 0.05; q = 1.0; RHO = 3.0
wrap = lambda x: (x + np.pi) % (2*np.pi) - np.pi
P0 = np.diag([-1, -1, -1, 1.5, 1.5]).astype(complex)
for _ in range(60000): P0 = P0 - 0.02*VG(P0, mu=0.3)
i, j = np.meshgrid(np.arange(L), np.arange(L), indexing='ij'); X = i + 0.5*j; Y = (np.sqrt(3)/2)*j
def rel(x0, y0):
    best = None
    for p in (-1, 0, 1):
        for qq in (-1, 0, 1):
            dx, dy = X - x0 + p*L + qq*L/2, Y - y0 + qq*np.sqrt(3)/2*L; r = np.hypot(dx, dy)
            if best is None: best = [r, dx, dy]
            else: k = r < best[0]; best = [np.where(k, r, best[0]), np.where(k, dx, best[1]), np.where(k, dy, best[2])]
    return best[1], best[2]
i0, j0 = 24, 16; cx, cy = float(X[i0, j0]), float(Y[i0, j0])           # v1
i1, j1 = i0 - 16, j0 + 32; vx2, vy2 = float(X[i1, j1]), float(Y[i1, j1])  # v2 над v1, відстань 27,7
wp = (cx + D, cy + 0.3)                     # +1 (зсув 0,3 по y, щоб не сісти точно на вузол)
wm = (cx + D + 20, cy + 0.3)                # −1 праворуч на 20
g = CGauge(np.full((L, L), -1.0), np.ones((3, L, L)), np.ones((2, L, L)), q=q, mu=0.3)
U = np.broadcast_to(np.eye(5), (3, L, L, 5, 5)).copy(); th = np.zeros((L, L)); h = np.ones((L, L))
if MODE in ('both', 'wind'):
    Mx, My = (wp[0] + wm[0])/2, wp[1]; mdx, mdy = rel(Mx, My)          # координати відносно середини пари: θ неперервна, вікно W
    for (x0, y0), sg in ((wp, 1), (wm, -1)):
        dx, dy = mdx - (x0 - Mx), mdy - (y0 - My); th += sg*np.arctan2(dy, dx); h *= 1 - np.exp(-(dx**2 + dy**2)/RHO**2)
    rM = np.hypot(mdx, mdy); th *= np.clip((24 - rM)/10, 0, 1)
    f = np.ones((L, L))
    for (x0, y0) in (wp, wm): dx, dy = rel(x0, y0); f *= np.tanh(np.hypot(dx, dy)/2)
else: f = np.ones((L, L))
P = (f*np.exp(1j*th))[..., None, None]*P0
A = np.array([-wrap(sh(th, d) - th)/q*0.5*(h + sh(h, d)) for d in DIRS])
ncut = 0
if MODE in ('both', 'alice'):
    Z = np.diag([1., 1., -1., -1., 1.]); xc_ = cx + 0.25
    for m, ev in enumerate([(1.0, 0.0), (0.5, np.sqrt(3)/2), (-0.5, np.sqrt(3)/2)]):
        s_ = (xc_ - X)/ev[0]; yc = Y + s_*ev[1]
        hit = (s_ > 0) & (s_ < 1) & (yc > cy) & (yc < vy2); U[m][hit] = Z; ncut += int(hit.sum())
P = sym0(P + 1e-3*np.random.default_rng(1).standard_normal(P.shape))
E = np.zeros_like(U); Pi = np.zeros_like(P); EA = np.zeros_like(A)
theta = lambda P: np.angle(np.trace(P @ P @ P, axis1=-2, axis2=-1)*np.conj(np.trace(P @ P, axis1=-2, axis2=-1)))
def bcent():
    Fu, Fd = g.flux(A); F = wrap(q*Fu) + wrap(q*Fd); dx, dy = rel(cx, cy); m = np.hypot(dx, dy) < 12
    w = np.abs(F)*m; s = w.sum()
    return (float((w*dx).sum()/s), float((w*dy).sum()/s), float(F[m].sum()/(2*np.pi))) if s > 1e-6 else (np.nan, np.nan, 0.0)
def ecent():                                  # ядро Аліси = центр потоку SO(5): Σ(5 − tr H) у трикутниках біля v1; + мін |Φ|²
    Hu, Hd = g.hol(U); w0 = (5 - np.trace(Hu, axis1=-2, axis2=-1)) + (5 - np.trace(Hd, axis1=-2, axis2=-1))
    mag = np.sum(np.abs(P)**2, axis=(-2, -1)); dx, dy = rel(cx, cy); m = np.hypot(dx, dy) < 12
    w = w0*m; s = w.sum(); return (float((w*dx).sum()/s), float((w*dy).sum()/s), float(mag[m].min())) if s > 1e-6 else (np.nan, np.nan, float(mag[m].min()))
out = dict(mode=MODE, D=D, ncut=ncut, t=[], E=[], bc=[], ec=[])
t = 0.0; nst = int(round(T/dt)); tic = time.time()
for s in range(nst + 1):
    if s % 40 == 0:
        out['t'].append(t); out['E'].append(float(g.edens(P, Pi, U, E, A, EA).sum())); out['bc'].append(bcent()); out['ec'].append(ecent())
        if s % 400 == 0: print(f't={t:.0f} E={out["E"][-1]:.3f} B(x,y,flux)={np.round(out["bc"][-1],2)} core(x,y,min)={np.round(out["ec"][-1],2)} ({time.time()-tic:.0f}s)', flush=True)
    if s == nst: break
    P, Pi, U, E, A, EA = g.step(P, Pi, U, E, A, EA, dt, gam=GAM if t < TG else 0.0); t += dt
json.dump(out, open(f'aw_{MODE}_D{D:g}.json', 'w')); np.savez_compressed(f'aw_{MODE}_D{D:g}_fin.npz', P=P, A=A, U=U)
