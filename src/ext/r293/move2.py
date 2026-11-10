"""(293) крок 2: «годинник» вихору в русі. Як r292/mass.py, але: ядро спершу збурено диханням (eps 0,1, як breathe.py br),
сила g діє лише TACC од., далі вихор летить сам (g = 0) ще T од.; пишемо tr P² (середнє r<2) і енергію r<3 навколо поточного центру.
python3 move2.py G TACC T
(292) Чи росте маса вихору зі швидкістю. Стала сила, що їде разом з вихором:
a(x) = −1 + g·dx·exp(−r²/2σ²) навколо кожного вихору (σ = 10): у центрі вихору глибина −1, нахил g → F = 0,921·g завжди.
Центр вікна оновлюється кожні 0,5 од. з лінійним прогнозом. Копія без вихорів з тим самим вікном — для енергії-надлишку.
Старт — r291/e0_a-1_L96.npz; 20 од. тертя 1 (вікно стоїть), далі T од. без тертя.
python3 mass.py G T [SIG]"""
import sys, os, json, time, numpy as np
sys.path.insert(0, '/mnt/project-files/model/s52/r268/code')
import gauge91 as G, phi90 as PH
gg, TACC, T = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]); SIG = 10.0
z = np.load('../r291/e0_a-1_L96.npz'); P, U = z['P'].copy(), z['U'].copy(); cx, cy, vy2 = float(z['cx']), float(z['cy']), float(z['vy2'])
L = 96; X, Y = PH.coords(L); dt = 0.05; b = 0.1
def wrapdx(px, py):      # мінімальний зсув X − px з урахуванням ромба
    best = None; bdx = None
    for p in (-1, 0, 1):
        for q in (-1, 0, 1):
            dx = X - px + p*L + q*L/2; dy = Y - py + q*np.sqrt(3)/2*L; r2 = dx**2 + dy**2
            if best is None: best, bdx = r2, dx
            else: m = r2 < best; best = np.where(m, r2, best); bdx = np.where(m, dx, bdx)
    return bdx, best
def amap(c):
    a = -np.ones((L, L))
    for (px, py) in c:
        dx, r2 = wrapdx(px, py); a += gg*dx*np.exp(-r2/(2*SIG**2))
    return a
g = G.Gauge(amap([(cx, cy), (cx, vy2)]), np.ones((3, L, L)), np.ones((2, L, L)), 1.0, 1.0, b)
P0 = np.diag(np.sort(np.linalg.eigvalsh(PH.vacuum(-1.0, b))))
Pv = np.broadcast_to(P0, P.shape).copy(); Uv = np.broadcast_to(np.eye(5), U.shape).copy()
S = [P, np.zeros_like(P), U, np.zeros_like(U)]; V = [Pv, np.zeros_like(Pv), Uv, np.zeros_like(Uv)]
def pos(c0):
    ex = g.edens(*S) - g.edens(*V); out = []; wid = []
    for (px, py) in c0:
        dx, r2 = wrapdx(px, py); w = np.clip(ex, 0, None)*(r2 < 25)
        dy = np.sqrt(np.maximum(r2 - dx**2, 0))*np.sign(Y - py)   # наближено
        mx = (w*dx).sum()/w.sum(); my = (w*dy).sum()/w.sum()
        wid.append((float(np.sqrt((w*(dx - mx)**2).sum()/w.sum())), float(np.sqrt((w*(dy - my)**2).sum()/w.sum()))))
        out.append(((px + mx) % L, py + my))
    return out, wid, float(ex.sum())
c = [(cx, cy), (cx, vy2)]; t = 0.0
while t < 20 - 1e-9:
    S = list(g.step(*S, dt, gam=1.0)); V = list(g.step(*V, dt, gam=1.0)); t += dt
c, wid, Ex = pos(c); c, wid, Ex = pos(c)
for (px, py) in c:
    r = PH.tdist(X, Y, px, py, L); S[0] = PH.sym0(S[0] + 0.1*(S[0] - P0)*np.exp(-r**2/8)[..., None, None])
S[1][:] = 0; S[3][:] = 0
c, wid, Ex = pos(c); g.a = amap(c)
tr2 = lambda P: np.einsum('...ij,...ji->...', P, P)
def sig(c):
    q = tr2(S[0]); e = g.edens(*S) - g.edens(*V); o = []
    for (px, py) in c:
        r = PH.tdist(X, Y, px, py, L); o.append((float(q[r < 2].mean()), float(e[r < 3].sum())))
    return o
rec = [dict(t=0.0, p=c, w=wid, E=Ex, s=sig(c))]; t = 0.0; t0 = time.time(); gg_on = True; vel = [0.0, 0.0]; tl = 0.0; cl = c; n = 0
while t < TACC + T - 1e-9:
    if gg and t >= TACC - 1e-9 and gg_on: g.a = -np.ones((L, L)); gg_on = False
    S = list(g.step(*S, dt, gam=0.0)); V = list(g.step(*V, dt, gam=0.0)); t += dt; n += 1
    if n % 2 == 0 and gg_on:   # прогноз центру вікна кожні 0,1
        g.a = amap([((px + vel[i]*(t - tl)) % L, py) for i, (px, py) in enumerate(cl)])
    if n % 5 == 0:
        c, wid, Ex = pos([((px + vel[i]*(t - tl)) % L, py) for i, (px, py) in enumerate(cl)])
        vel = [((c[i][0] - cl[i][0] + L/2) % L - L/2)/(t - tl) for i in range(2)]; tl = t; cl = c
        g.a = amap(c) if gg_on else g.a
        rec.append(dict(t=round(t, 2), p=c, w=wid, E=Ex, v=vel, s=sig(c)))
        if n % 400 == 0: print(rec[-1], round(time.time() - t0), flush=True)
        json.dump(dict(g=gg, tacc=TACC, sig=SIG, T=T, rec=rec), open(f'mv_g{gg:g}_acc{TACC:g}.json', 'w'))
