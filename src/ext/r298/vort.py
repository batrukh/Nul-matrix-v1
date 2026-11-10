"""(298) етап 2: вихор із цілим числом обертів фази Φ★ у SO(5)×U(1).
MODE: pair (±1 на відстані D), ctrl (ті самі ядра без обертів), alice (пара Аліси з r294, a≡0).
Міра: калібрувально-інваріантне число n = (1/2π)(Σ_∂ χ − qΣ_∂ a), χ = wrap(θ(n+d) − θ(n) + q a), θ = arg(trΦ³·conj trΦ²), — у квадраті навколо кожного ядра; потік B = Σ wrap(F)/2π; енергія; |Φ|² у ядрі."""
import numpy as np, sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
from cgauge import CGauge, sym0
from gauge91 import DIRS, sh
from vac import G as VG
MODE = sys.argv[1]; D = int(sys.argv[2]); T = float(sys.argv[3]); GAM = float(os.environ.get('GAM', '0.2')); TG = float(os.environ.get('TG', '100'))
L = int(os.environ.get('L', '64')); dt = 0.05; q = 1.0
wrap = lambda x: (x + np.pi) % (2*np.pi) - np.pi
P0 = np.diag([-1, -1, -1, 1.5, 1.5]).astype(complex)
for _ in range(60000): P0 = P0 - 0.02*VG(P0, mu=0.3)
i, j = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
X = i + 0.5*j; Y = (np.sqrt(3)/2)*j                                  # трикутна ґратка: e0=(1,0), e1=(½,√3/2)
def rel(x0, y0):
    dx = X - x0; dy = Y - y0
    dy = dy - L*np.sqrt(3)/2*np.round(dy/(L*np.sqrt(3)/2)); dx = dx - 0.5*L*np.round((Y - y0 - dy)/(L*np.sqrt(3)/2))
    dx = dx - L*np.round(dx/L); return dx, dy
yc = L*np.sqrt(3)/4; xc = L/2 + 0.5*(L/2)
cores = [(xc - D/2, yc), (xc + D/2, yc)]
SG = [1, -1]
if MODE == 'quad':                                                   # ++ поруч (D), −− на півтора поля далі
    cores = [(xc - D/2, yc), (xc + D/2, yc), (xc - D/2 + L/4, yc + L*np.sqrt(3)/4), (xc + D/2 + L/4, yc + L*np.sqrt(3)/4)]; SG = [1, 1, -1, -1]
wl = np.ones((3, L, L)); wt = np.ones((2, L, L)); g = CGauge(np.full((L, L), -1.0), wl, wt, q=q, mu=0.3)
th = np.zeros((L, L)); f = np.ones((L, L))
for k, (x0, y0) in enumerate(cores):
    dx, dy = rel(x0, y0); r = np.hypot(dx, dy); f *= np.tanh(r/2.0)
    if MODE in ('pair', 'quad'): th += SG[k]*np.arctan2(dy, dx)
U = np.broadcast_to(np.eye(5), (3, L, L, 5, 5)).copy(); E = np.zeros_like(U)
if MODE == 'alice':
    st = np.load(os.path.dirname(os.path.abspath(__file__)) + '/../r294/e0_ka1_kb2_a-1_L96.npz'); P = st['P'].astype(complex); U = st['U'].copy(); E = np.zeros_like(U)
    L = P.shape[0]; wl = np.ones((3, L, L)); wt = np.ones((2, L, L)); g = CGauge(np.full((L, L), -1.0), wl, wt, q=q, mu=0.3); A = np.zeros((3, L, L))
    i, j = np.meshgrid(np.arange(L), np.arange(L), indexing='ij'); X = i + 0.5*j; Y = (np.sqrt(3)/2)*j
    cores = [(float(st['cx']) + 0.5*float(v), np.sqrt(3)/2*float(v)) for v in np.atleast_1d(st['cy'])]
else:
    P = (f*np.exp(1j*th))[..., None, None]*P0
    A = np.array([wrap(th - sh(th, d))/q*0 + wrap(sh(th, d) - th)*0 for d in DIRS])   # спершу a=0, далі — чиста калібровка
    A = np.array([-wrap(sh(th, d) - th)/q for d in DIRS])
    RHO = float(os.environ.get('RHO', '0'))                         # RHO>0: потік розмазаний (профіль Нільсена–Олесена), 0: весь потік 2π в одному трикутнику
    if RHO > 0:
        h = np.ones((L, L))
        for (x0, y0) in cores:
            dx, dy = rel(x0, y0); h *= 1 - np.exp(-(dx**2 + dy**2)/RHO**2)
        A = np.array([A[m]*0.5*(h + sh(h, d)) for m, d in enumerate(DIRS)])
    # e^{iqa}Φ(n+d) ≈ Φ(n): q a = θ(n) − θ(n+d)
EA = np.zeros_like(A); Pi = np.zeros_like(P)
def theta(P):
    return np.angle(np.trace(P @ P @ P, axis1=-2, axis2=-1)*np.conj(np.trace(P @ P, axis1=-2, axis2=-1)))
RW = int(os.environ.get('RW', '6'))
def wind(P, A, x0, y0, R=None):
    R = R or RW
    dx, dy = rel(x0, y0); ic = np.unravel_index(np.argmin(dx**2 + dy**2), dx.shape)
    tt = theta(P); tot = 0.0; ch = 0.0
    # квадратна петля в індексах (i,j): вздовж e0, e1, −e0, −e1
    path = [(0, +1)]*(2*R) + [(1, +1)]*(2*R) + [(0, -1)]*(2*R) + [(1, -1)]*(2*R)
    a0, b0 = (ic[0] - R) % L, (ic[1] - R) % L
    for m, sgn in path:
        d = DIRS[m]
        if sgn > 0: n0 = (a0, b0); n1 = ((a0 + d[0]) % L, (b0 + d[1]) % L); aa = A[m][n0]
        else: n1 = ((a0 - d[0]) % L, (b0 - d[1]) % L); n0 = n1; aa = A[m][n1]
        if sgn > 0: chi = wrap(tt[n1] - tt[n0] + q*aa); ch += chi; tot += q*aa; a0, b0 = n1
        else: chi = wrap(tt[(a0, b0)] - tt[n1] + q*aa); ch -= chi; tot -= q*aa; a0, b0 = n1
    return (ch - tot)/(2*np.pi)
def fluxbox(A, x0, y0, R=None):
    R = R or RW
    Fu, Fd = g.flux(A); dx, dy = rel(x0, y0); m = (np.abs(dx) < R) & (np.abs(dy) < R)
    return (wrap(q*Fu)[m].sum() + wrap(q*Fd)[m].sum())/(2*np.pi)
out = {'mode': MODE, 'D': D, 'L': L, 't': [], 'E': [], 'n': [], 'B': [], 'core': []}
t = 0.0; tic = time.time(); nst = int(round(T/dt))
for s in range(nst + 1):
    if s % int(round(2/dt)) == 0:
        e = g.edens(P, Pi, U, E, A, EA); out['t'].append(t); out['E'].append(float(e.sum()))
        out['n'].append([float(wind(P, A, *c)) for c in cores]); out['B'].append([float(fluxbox(A, *c)) for c in cores])
        mag = np.sum(np.abs(P)**2, axis=(-2, -1)); out['core'].append([float(mag[np.unravel_index(np.argmin(sum(v**2 for v in rel(*c))), mag.shape)]) for c in cores])
        if s % int(round(20/dt)) == 0: print(f't={t:.0f} E={out["E"][-1]:.3f} n={np.round(out["n"][-1],3)} B={np.round(out["B"][-1],3)} core={np.round(out["core"][-1],3)} ({time.time()-tic:.0f}s)', flush=True)
    if s == nst: break
    P, Pi, U, E, A, EA = g.step(P, Pi, U, E, A, EA, dt, gam=GAM if t < TG else 0.0); t += dt
json.dump(out, open(f'vort_{MODE}_D{D}{os.environ.get("TAG","")}.json', 'w'))
np.savez_compressed(f'vort_{MODE}_D{D}{os.environ.get("TAG","")}_fin.npz', P=P, A=A, U=U)
