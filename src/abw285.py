"""(285) Розсіяння «фотона» на Z₂-вихорі (струна Аліси) в однорідному світі 3+2.
Синтетичний світ: χ=1 скрізь (ваги 1), a=−1, b=0,1, вакуум Φ★ = diag(−λ1×3, λ2×2) після релаксації. ψ і нитки не рахуються (як (145), (282)).
Вихор v1 у центрі, партнер v2 на відстані половини ґратчастого вектора (L/4, √3L/4) — розріз між ними поглинає калібрування ланок (як plant_short).
Хвиля — пакет поперечного потенціалу A = ŷ·cos(k(X−x0))·exp(−(X−x0)²/2σx²)·exp(−(Y−cy)²/2σy²) у локальній площині «фотона» gg·(e3,e4),
з E = −c·∂ₓθ (біжить у +X). Дві копії (з поштовхом і без) — різниця = хвиля. Міряємо кінетичну енергію різниці k(x,t) і її суму за час (флюенс).
python3 abw.py MODE(vort|ctrl) LAM [L] [T]"""
import os as _o; _V = _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), 'variant'); RUNS = _o.environ.get('NM_RUNS', _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), '..', 'runs', 'out', 'variant'))
import sys, os, json, time, numpy as np
sys.path.insert(0, _V)
import gauge91 as G, phi90 as PH
mode, LAM = sys.argv[1], float(sys.argv[2]); L = int(sys.argv[3]) if len(sys.argv) > 3 else 128; T = float(sys.argv[4]) if len(sys.argv) > 4 else 60.0
C = float(os.environ.get('CPH', '0.65')); AMP = float(os.environ.get('AMP', '0.02')); SX = float(os.environ.get('SX', str(LAM))); SY = float(os.environ.get('SY', '12'))
X0OFF = float(os.environ.get('X0', '-28')); Tr = float(os.environ.get('TR', '60')); dt = 0.05; b = 0.1
X, Y = PH.coords(L); a = np.full((L, L), -1.0); wl, wt = np.ones((3, L, L)), np.ones((2, L, L))
g = G.Gauge(a, wl, wt, 1.0, 1.0, b)
P0 = np.diag([-0.3485, -0.3485, -0.3485, 0.5228, 0.5228])
I5 = np.eye(5)
def wrapd(cx, cy, x=X, y=Y):
    best = None
    for p in (-1, 0, 1):
        for q in (-1, 0, 1):
            dx, dy = x - cx + p*L + q*L/2, y - cy + q*np.sqrt(3)/2*L; r = np.hypot(dx, dy)
            if best is None: best = [r, dx, dy]
            else: k = r < best[0]; best = [np.where(k, r, best[0]), np.where(k, dx, best[1]), np.where(k, dy, best[2])]
    return best
i0, j0 = L//2 - L//8, L//2           # v1 (центр по Y)
cx, cy = float(X[i0, j0]), float(Y[i0, j0])
i1, j1 = i0 - 20, j0 + 40                 # v2: над v1 (той самий X), відстань 40·√3/2 ≈ 34,6
vx2, vy2 = float(X[i1, j1]), float(Y[i1, j1])
P = np.broadcast_to(P0, (L, L, 5, 5)).copy(); U = np.broadcast_to(I5, (3, L, L, 5, 5)).copy()
gg = np.broadcast_to(I5, (L, L, 5, 5)).copy()
Zpi = np.diag([1.0, 1.0, -1.0, -1.0, 1.0])   # R(π) у площині (e2, e3): лежить у незламаній групі, але в її неспряженій компоненті
EV = [(1.0, 0.0), (0.5, np.sqrt(3)/2), (-0.5, np.sqrt(3)/2)]
ncut = 0
if mode == 'vort':
    # калібрувальна посадка: Φ★ однорідне, ланки, що перетинають вертикальний відрізок X = cx+0,25 між v1 і v2, несуть R(π).
    # Голономія навколо v1 (і v2) = R(π): площина «фотона» (e3,e4) обертається → струна Аліси. Потік лише в плакетах на кінцях.
    xc_ = cx + 0.25
    for m, ev in enumerate(EV):
        if abs(ev[0]) < 1e-9: continue
        s_ = (xc_ - X)/ev[0]                  # параметр перетину вздовж ланки n → n+e
        yc = Y + s_*ev[1]
        hit = (s_ > 0) & (s_ < 1) & (yc > cy) & (yc < vy2)
        U[m][hit] = Zpi; ncut += int(hit.sum())
    rng = np.random.default_rng(1)
    P = PH.sym0(P + 1e-3*rng.standard_normal(P.shape))   # малий шум, щоб потік міг розповзтися під час релаксації
E = np.zeros_like(U); Pi = np.zeros_like(P)
t = 0.0
while t < Tr - 1e-9: P, Pi, U, E = g.step(P, Pi, U, E, dt, gam=1.0); t += dt
Pi[:] = 0; E[:] = 0
P_rel = P.copy(); e_rel = g.edens(P, Pi, U, E); U_rel = U.copy()
G0 = np.zeros((5, 5)); G0[3, 4], G0[4, 3] = 1, -1
Gk = gg @ G0 @ np.swapaxes(gg, -1, -2)             # локальна площина «фотона» (розріз далеко від джерела)
x0 = cx + X0OFF; k = 2*np.pi/LAM
def theta(m, shift):
    ev = EV[m]; dx, dy = X + ev[0]/2 - (x0 + shift), Y + ev[1]/2 - cy   # без періодичних образів (джерело далеко від країв ромба)
    return AMP*np.cos(k*dx)*np.exp(-dx**2/(2*SX**2))*np.exp(-dy**2/(2*SY**2))*ev[1]
UB = U.copy(); EB = E.copy(); h = 1e-3
for m in range(3):
    th = theta(m, 0.0); thd = (theta(m, C*h) - th)/h   # θ(X − c t): ∂ₜθ = c·∂θ/∂x0
    UB[m] = G.cayley(th[..., None, None]*Gk) @ U[m]
    EB[m] = thd[..., None, None]*Gk
PA, PiA, UA, EA = P.copy(), Pi.copy(), U.copy(), E.copy(); PB, PiB = P.copy(), Pi.copy()
fl = np.zeros((L, L)); snaps = []; ts = []; sig = []; tsig = []; t = 0.0; t0 = time.time()
G0n = G0/np.sqrt((G0**2).sum())
while t < T - 1e-9:
    PA, PiA, UA, EA = g.step(PA, PiA, UA, EA, dt); PB, PiB, UB, EB = g.step(PB, PiB, UB, EB, dt); t += dt
    dE_ = EB - EA; dPi = PiB - PiA
    kin = 0.5*np.sum(dPi**2, axis=(-2, -1)) + sum(0.25*np.sum(dE_[m]**2, axis=(-2, -1)) for m in range(3))
    fl += kin*dt
    if abs(t*2 - round(t*2)) < dt/2:   # знаковий «фотон»: проєкція δE на G0 (калібрування, де розріз лише над v1), поляризація y
        sig.append(sum(np.sum(dE_[m]*G0n, axis=(-2, -1))*EV[m][1] for m in range(3)).astype(np.float32)); tsig.append(round(t, 2))
    if abs(t - round(t)) < dt/2 and round(t) % 2 == 0: snaps.append(kin.astype(np.float32)); ts.append(round(t, 2))
out = f'abw_{mode}_L{L}_lam{LAM:g}_c{C:g}'
np.savez(out + '_U.npz', U=U_rel.astype(np.float32)) if T < 1 else None
np.savez(out + '.npz', P=P_rel.astype(np.float32), e_rel=e_rel.astype(np.float32), fl=fl, sig=np.array(sig), tsig=np.array(tsig), k=np.array(snaps), t=np.array(ts), X=X, Y=Y)
json.dump(dict(mode=mode, lam=LAM, L=L, T=T, C=C, amp=AMP, sx=SX, sy=SY, x0=x0, v1=[cx, cy], v2=[vx2, vy2], ncut=ncut, sec=time.time() - t0), open(out + '.json', 'w'))
print(out, 'v1', (cx, cy), 'v2', (vx2, vy2), 'sec', round(time.time() - t0))
