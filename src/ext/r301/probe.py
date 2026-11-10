"""(301) Чи тримає «заряджена Аліса» (складений вихор (300)) непарний квант. Модель SO(5)×U(1) (r298/cgauge.py).
MODE: comp (стан aw_both_D8_fin: Аліса + оберт +1 злиплись), alice (aw_alice_D0_fin: лише Аліса), vac (однорідний вакуум, k=0).
Поштовх лише непарних щодо Z компонент Φ (дійсна й уявна частини), гаус r0=3 у ядрі; без поштовху E (закон Гаусса); поглинальний шар r>16.
EPS=0 — фон (той самий стан без поштовху). Запис Φ у r<7 кожні 0,5.  python3 probe.py MODE EPS T"""
import sys, os, time, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H + '/../r298'); sys.path.insert(0, H + '/..')
from cgauge import CGauge, sym0
from vac import G as VG
MODE, EPS, T = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]); L = 64; dt = 0.05
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
if MODE in ('vac', 'vacloc'):
    P = np.broadcast_to(P0, (L, L, 5, 5)).copy(); U = np.broadcast_to(np.eye(5), (3, L, L, 5, 5)).copy(); A = np.zeros((3, L, L)); cx, cy = 32., 27.7
else:
    z = np.load(H + f'/../r300/aw_{"both_D8" if MODE == "comp" else "alice_D0"}_fin.npz'); P, U, A = z['P'].copy(), z['U'].copy(), z['A'].copy()
    Hu, Hd = g.hol(U); w0 = np.real((5 - np.trace(Hu, axis1=-2, axis2=-1)) + (5 - np.trace(Hd, axis1=-2, axis2=-1)))
    r0, dx0, dy0 = rel(32.0, 13.856); guess = (32.0 + (6.0 if MODE == 'comp' else 0.0), 13.856)
    r, dx, dy = rel(*guess); m = r < 6; s = (w0*m).sum(); cx, cy = guess[0] + (w0*m*dx).sum()/s, guess[1] + (w0*m*dy).sum()/s
r, dx, dy = rel(cx, cy); f = np.exp(-r**2/(2*3.0**2)) if MODE != 'vac' else np.ones((L, L))   # vacloc: той самий локальний поштовх у вакуумі
zs = np.array([1, 1, -1, -1, 1.0]); Zm = np.outer(zs, zs) < 0
rng = np.random.default_rng(7); K = rng.standard_normal((L, L, 5, 5)) + 1j*rng.standard_normal((L, L, 5, 5))
K = 0.5*(K + np.swapaxes(K, -1, -2))*Zm; P = sym0(P + EPS*K*f[..., None, None])
Pi = np.zeros_like(P); E = np.zeros_like(U); EA = np.zeros_like(A)
gam = 0.3*np.clip((r - 16)/6, 0, 1)**2; gamP = gam[..., None, None]; gamE = gam[None, ..., None, None]; gamA = gam[None]
from gauge91 import cayley
iu = np.triu_indices(5); M = r < 7
rec = []; t = 0.0; n = 0; t0 = time.time()
while t < T - 1e-9:
    if n % 10 == 0:
        dP = (P - P0)[..., iu[0], iu[1]]
        rec.append(dP[M].astype(np.complex64) if MODE != 'vac' else dP.mean((0, 1)).astype(np.complex64))
    # тертя лише в поглинальному шарі r>16 (Π, E, EA)
    FP, FE, FEA = g.forces(P, U, A)
    Pi = Pi + 0.5*dt*(FP - gamP*Pi); E = E + 0.5*dt*(FE - gamE*E); EA = EA + 0.5*dt*(FEA - gamA*EA)
    P = sym0(P + dt*Pi); U = cayley(dt*E) @ U; A = A + dt*EA
    FP, FE, FEA = g.forces(P, U, A)
    Pi = (Pi + 0.5*dt*FP)/(1 + 0.5*dt*gamP); E = (E + 0.5*dt*FE)/(1 + 0.5*dt*gamE); EA = (EA + 0.5*dt*FEA)/(1 + 0.5*dt*gamA)
    t += dt; n += 1
    if n % 2000 == 0: print(round(t), round(time.time() - t0), flush=True)
np.savez(f'probe_{MODE}' + ('_bg' if EPS == 0 else '') + '.npz', rec=np.array(rec), X=X[M] if MODE != 'vac' else 0, Y=Y[M] if MODE != 'vac' else 0,
         dx=dx[M] if MODE != 'vac' else 0, dy=dy[M] if MODE != 'vac' else 0, cx=cx, cy=cy, dts=dt*10)
print('done', MODE, EPS, 'core', round(cx, 2), round(cy, 2), round(time.time() - t0))
