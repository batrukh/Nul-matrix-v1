"""(312) спільна релаксація вихору бази r294 (L96) і директора. python3 relax312.py MU K TAG [VAC]
Етап 1: директор пробно (Φ, U заморожені) зі старту — перенесений вздовж дерева BFS загальний вектор трійки (без упередження до e0/e2).
Етап 2: усе разом, загасна динаміка (γ 0,5, dt 0,1) до max|сили| < 1e-6. Контрольні точки в OUT кожні 2000 кроків."""
import numpy as np, sys, os, time, json
sys.path.insert(0, '/tmp/claude-0/-home-user/11ca481a-2b4a-53b5-a097-dab19d222dd0/scratchpad/r312'); from gd import *
sys.path.insert(0, '/mnt/project-files/model/s52/r268/code'); import phi90 as PH
MU, K, TAG = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]; VAC = len(sys.argv) > 4
OUT = f'/mnt/project-files/model/s52/r312/ck_{TAG}.npz'; os.makedirs(os.path.dirname(OUT), exist_ok=True)
z = np.load('/mnt/project-files/model/s52/r294/e0_ka1_kb2_a-1_L96.npz'); L = 96
P, U = z['P'].copy(), z['U'].copy()
if VAC:
    Pv = np.diag(np.sort(np.linalg.eigvalsh(PH.vacuum(-1.0, 0.1)))); L = 16
    P = np.broadcast_to(Pv, (L, L, 5, 5)).copy(); U = np.broadcast_to(np.eye(5), (3, L, L, 5, 5)).copy()
g = GD(-np.ones((L, L)), np.ones((3, L, L)), np.ones((2, L, L)), 1.0, 2.0, 0.1, mu=MU, k=K)
if os.path.exists(OUT):
    c = np.load(OUT); P, Pi, U, E, n, pn, stage, it0 = c['P'], c['Pi'], c['U'], c['E'], c['n'], c['pn'], int(c['stage']), int(c['it'])
    print('resume', stage, it0, flush=True)
else:
    rng = np.random.default_rng(3); ref = (int(z['cx']) + L//2) % L if not VAC else 0
    ref = (ref, int(z['cy']/(np.sqrt(3)/2))) if not VAC else (0, 0)
    w, V = np.linalg.eigh(P[ref]); v = V[:, :3] @ rng.normal(size=3); v /= np.linalg.norm(v)
    n = np.zeros((L, L, 5)); seen = np.zeros((L, L), bool); n[ref] = v; seen[ref] = True; q = [ref]
    while q:
        a = q.pop(0)
        for m, d in enumerate(DIRS):
            for sg in (1, -1):
                b = ((a[0] + sg*d[0]) % L, (a[1] + sg*d[1]) % L)
                if seen[b]: continue
                n[b] = (U[m][a].T @ n[a]) if sg == 1 else (U[m][b] @ n[a]); seen[b] = True; q.append(b)
    if os.environ.get('INIT') == 'least':                             # старт: вектор трійки з найменшим |власним значенням| Φ (кандидат e2)
        w_, V_ = np.linalg.eigh(P); n = V_[..., :, 2]
    n /= np.linalg.norm(n, axis=-1, keepdims=True); Pi = np.zeros_like(P); E = np.zeros_like(U); pn = np.zeros_like(n); stage, it0 = 1, 0
tang = lambda v, n: v - np.sum(v*n, -1, keepdims=True)*n
def tot(P, U, n): return float(np.sum(g.edens(P, np.zeros_like(P), U, np.zeros_like(U))) + np.sum(g.en(P, U, n)))
t0 = time.time(); dt, gam = 0.1, 0.5
if stage == 1:
    vel = pn
    for it in range(it0, int(os.environ.get('NPROBE', 60000)) + 1):
        _, _, Fn = g.forces_d(P, U, n); gt = tang(Fn, n); gn = np.abs(gt).max()
        if it % 2000 == 0:
            print('1', it, f'{tot(P, U, n):.8f}', f'{gn:.2e}', round(time.time() - t0), flush=True)
            np.savez(OUT, P=P, Pi=Pi, U=U, E=E, n=n, pn=vel, stage=1, it=it)
        if gn < 1e-7: break
        vel = 0.9*vel + 0.1*gt; vel = tang(vel, n)
        if np.sum(vel*gt) < 0: vel[:] = 0
        n = n + vel; n /= np.linalg.norm(n, axis=-1, keepdims=True)
    np.savez(OUT.replace('ck_', 'probe_'), P=P, U=U, n=n); stage, it0, pn = 2, 0, np.zeros_like(n)
for it in range(it0, 200001):
    P, Pi, U, E, n, pn = g.step_d(P, Pi, U, E, n, pn, dt, gam)
    if it % 2000 == 0:
        FP, FE, Fn = g.forces_d(P, U, n); fm = max(np.abs(FP).max(), np.abs(FE).max(), np.abs(tang(Fn, n)).max())
        print('2', it, f'{tot(P, U, n):.8f}', f'{fm:.2e}', round(time.time() - t0), flush=True)
        np.savez(OUT, P=P, Pi=Pi, U=U, E=E, n=n, pn=pn, stage=2, it=it)
        if fm < 1e-6: break
np.savez(OUT.replace('ck_', 'fin_'), P=P, U=U, n=n, E=tot(P, U, n), cx=z['cx'], cy=z['cy'])
print('done', tot(P, U, n), flush=True)
