"""(291) крок 2: вихор Аліси в нахилі потенціалу a(x) = −1 + g·(L/2π)·sin(2π(X − x_v)/L) (у вихорі нахил = g).
Старт — релаксований стан e0_a-1_L96.npz; 20 од. тертя 1 з нахилом (фон підлаштовується), далі T од. з тертям GAM (0 — інерція, 0,2 — як у базі).
Копія без вихорів (той самий нахил) — для енергії-надлишку; положення = центроїд надлишку енергії в колі r < 5 навколо кожного вихору.
python3 move.py G GAM [T]"""
import sys, json, time, numpy as np
sys.path.insert(0, '/mnt/project-files/model/s52/r268/code')
import gauge91 as G, phi90 as PH
import os; KA = float(os.environ.get("KA", "0.5")); KB = float(os.environ.get("KB", "1")); ST = os.environ.get("START", "../r291/e0_a-1_L96.npz")
gg, GAM = float(sys.argv[1]), float(sys.argv[2]); T = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
z = np.load(ST); P, U = z['P'].copy(), z['U'].copy(); cx, cy, vy2 = float(z['cx']), float(z['cy']), float(z['vy2'])
L = 96; X, Y = PH.coords(L); dt = 0.05; b = 0.1
a = -1.0 + gg*L/(2*np.pi)*np.sin(2*np.pi*(X - cx)/L)
g = G.Gauge(a, np.ones((3, L, L)), np.ones((2, L, L)), KA, KB, b)
P0 = np.diag(np.sort(np.linalg.eigvalsh(PH.vacuum(-1.0, b))))
Pv = np.broadcast_to(P0, P.shape).copy(); Uv = np.broadcast_to(np.eye(5), U.shape).copy()
S = [P, np.zeros_like(P), U, np.zeros_like(U)]; V = [Pv, np.zeros_like(Pv), Uv, np.zeros_like(Uv)]
def pos(S, V, c0):
    ex = g.edens(*S) - g.edens(*V); out = []
    for (px, py) in c0:
        r = PH.tdist(X, Y, px, py, L); w = np.clip(ex, 0, None)*(r < 5)
        dx = (X - px + L/2) % L - L/2                    # без урахування зсуву ромба по y (рух уздовж x, |dy| < 5)
        out.append((float(px + (w*dx).sum()/w.sum()), float(py + (w*(Y - py)).sum()/w.sum())))
    return out
c = [(cx, cy), (cx, vy2)]; t = 0.0
while t < 20 - 1e-9:
    S = list(g.step(*S, dt, gam=1.0)); V = list(g.step(*V, dt, gam=1.0)); t += dt
c = pos(S, V, c); c = pos(S, V, c); rec = [dict(t=0.0, p=c)]; t = 0.0; t0 = time.time()
while t < T - 1e-9:
    S = list(g.step(*S, dt, gam=GAM)); V = list(g.step(*V, dt, gam=GAM)); t += dt
    if abs(t - round(t)) < dt/2:
        c = pos(S, V, c); rec.append(dict(t=round(t, 1), p=c))
        if round(t) % 10 == 0: print(rec[-1], round(time.time() - t0), flush=True)
json.dump(dict(g=gg, gam=GAM, T=T, rec=rec), open(f'move_ka{KA:g}_kb{KB:g}_g{gg:g}_gam{GAM:g}.json', 'w'))
