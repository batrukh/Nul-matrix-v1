"""(291) крок 1: енергія ядра вихору Аліси E0(a) у синтетичному однорідному світі (як (285), без h).
Пара вихорів калібрувально (відстань 34,6), релаксація з тертям; E0 = (E_вихори − E_без вихорів)/2. python3 e0.py A [L TR]"""
import sys, json, numpy as np
sys.path.insert(0, '/mnt/project-files/model/s52/r268/code')
import gauge91 as G, phi90 as PH
A = float(sys.argv[1]); L = int(sys.argv[2]) if len(sys.argv) > 2 else 96; TR = float(sys.argv[3]) if len(sys.argv) > 3 else 120
X, Y = PH.coords(L); b = 0.1; dt = 0.05
g = G.Gauge(np.full((L, L), A), np.ones((3, L, L)), np.ones((2, L, L)), 1.0, 1.0, b)
P0 = np.diag(np.sort(np.linalg.eigvalsh(PH.vacuum(A, b))))
i0, j0 = L//2 - L//8, L//2 - 20; cx, cy = float(X[i0, j0]), float(Y[i0, j0]); vy2 = float(Y[i0 - 20, j0 + 40])
Zpi = np.diag([1.0, 1.0, -1.0, -1.0, 1.0]); EV = [(1.0, 0.0), (0.5, np.sqrt(3)/2), (-0.5, np.sqrt(3)/2)]
out = {}
for mode in ('vort', 'vac'):
    P = np.broadcast_to(P0, (L, L, 5, 5)).copy(); U = np.broadcast_to(np.eye(5), (3, L, L, 5, 5)).copy()
    if mode == 'vort':
        for m, ev in enumerate(EV):
            s_ = (cx + 0.25 - X)/ev[0]; yc = Y + s_*ev[1]
            U[m][(s_ > 0) & (s_ < 1) & (yc > cy) & (yc < vy2)] = Zpi
        P = PH.sym0(P + 1e-3*np.random.default_rng(1).standard_normal(P.shape))
    Pi = np.zeros_like(P); E = np.zeros_like(U); t = 0.0; hist = []
    while t < TR - 1e-9:
        P, Pi, U, E = g.step(P, Pi, U, E, dt, gam=1.0); t += dt
        if abs(t/20 - round(t/20)) < dt/10: hist.append(float(g.edens(P, Pi, U, E).sum()))
    out[mode] = hist
    if mode == 'vort': np.savez(f'e0_a{A:g}_L{L}.npz', P=P, U=U, cx=cx, cy=cy, vy2=vy2)
E0 = [(v - w)/2 for v, w in zip(out['vort'], out['vac'])]
json.dump(dict(a=A, L=L, E0=E0, P0=np.diag(P0).tolist()), open(f'e0_a{A:g}_L{L}.json', 'w'))
print(A, 'E0 history', [round(e, 4) for e in E0], 'vac', np.round(np.diag(P0), 4))
