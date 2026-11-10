"""(293) крок 1: власні коливання ядра вихору Аліси в спокої.
Старт — r291/e0_a-1_L96.npz (релаксована пара). Збурення ядра: P → P + eps·(P − Pvac)·f(r), f = exp(−r²/2·2²) (MODE br — дихання)
або f·cos2θ (MODE q — квадруполь). Без тертя, T од. Сигнали щокроку 0,25: енергія-надлишок у r<3, tr P² у найближчому вузлі, середнє tr P² у r<2.
python3 breathe.py MODE EPS T"""
import sys, json, time, numpy as np
sys.path.insert(0, '/mnt/project-files/model/s52/r268/code')
import gauge91 as G, phi90 as PH
import os; KA = float(os.environ.get("KA", "0.5")); KB = float(os.environ.get("KB", "1")); ST = os.environ.get("START", "../r291/e0_a-1_L96.npz")   # (294) вага кінетичної енергії поля ниток κ_a (база 1; 0,5 — фотон c = 1)
MODE, EPS, T = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
z = np.load(ST); P, U = z['P'].copy(), z['U'].copy(); cx, cy, vy2 = float(z['cx']), float(z['cy']), float(z['vy2'])
L = 96; X, Y = PH.coords(L); dt = 0.05; b = 0.1
g = G.Gauge(-np.ones((L, L)), np.ones((3, L, L)), np.ones((2, L, L)), KA, KB, b)
P0 = np.diag(np.sort(np.linalg.eigvalsh(PH.vacuum(-1.0, b))))
Pv = np.broadcast_to(P0, P.shape).copy(); Uv = np.broadcast_to(np.eye(5), U.shape).copy()
ev = float(g.edens(Pv, np.zeros_like(Pv), Uv, np.zeros_like(Uv)).mean())
cen = [(cx, cy), (cx, vy2)]; R = [PH.tdist(X, Y, px, py, L) for px, py in cen]
for (px, py), r in zip(cen, R):
    f = np.exp(-r**2/8)
    if MODE == 'q':
        dx = (X - px + L/2) % L - L/2; dy = Y - py; f = f*np.cos(2*np.arctan2(dy, dx))
    P = PH.sym0(P + EPS*(P - P0)*f[..., None, None])
Pi = np.zeros_like(P); E = np.zeros_like(U)
tr2 = lambda P: np.einsum('...ij,...ji->...', P, P)
k0 = [np.unravel_index(np.argmin(r), r.shape) for r in R]
rec = []; t = 0.0; n = 0; t0 = time.time()
while t < T - 1e-9:
    if n % 5 == 0:
        e = g.edens(P, Pi, U, E) - ev; q = tr2(P)
        rec.append(dict(t=round(t, 3), E3=[float(e[r < 3].sum()) for r in R], c=[float(q[k]) for k in k0], m2=[float(q[r < 2].mean()) for r in R]))
    P, Pi, U, E = g.step(P, Pi, U, E, dt, gam=0.0); t += dt; n += 1
    if n % 1000 == 0:
        print(round(t), round(time.time() - t0), flush=True); json.dump(dict(mode=MODE, eps=EPS, rec=rec), open(f'br_ka{KA:g}_kb{KB:g}_{MODE}_{EPS:g}.json', 'w'))
json.dump(dict(mode=MODE, eps=EPS, rec=rec), open(f'br_ka{KA:g}_kb{KB:g}_{MODE}_{EPS:g}.json', 'w'))
