"""(294) крок 0: швидкості хвиль у вакуумі 3+2 (a −1, b 0,1), L48, шум 1e-3, без тертя.
Фотон: E34 (сума трьох ланок). Хвилі Φ★: компонента P01 (у трійці, безмасова?) і P34; моди (n,0) і (2n,n).
python3 disp.py KA KB [T]"""
import sys, numpy as np
sys.path.insert(0, '/mnt/project-files/model/s52/r268/code')
import gauge91 as G, phi90 as PH
KA, KB = float(sys.argv[1]), float(sys.argv[2]); T = float(sys.argv[3]) if len(sys.argv) > 3 else 300.0
L, dt, b = 48, 0.05, 0.1
g = G.Gauge(-np.ones((L, L)), np.ones((3, L, L)), np.ones((2, L, L)), KA, KB, b)
P0 = np.diag(np.sort(np.linalg.eigvalsh(PH.vacuum(-1.0, b))))
P = np.broadcast_to(P0, (L, L, 5, 5)).copy(); U = np.broadcast_to(np.eye(5), (3, L, L, 5, 5)).copy()
rng = np.random.default_rng(11); A = 1e-3
Pi = PH.sym0(A*rng.standard_normal(P.shape)); E = A*rng.standard_normal(U.shape); E = 0.5*(E - np.swapaxes(E, -1, -2))
I, J = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
modes = [(n, 0) for n in range(1, 7)] + [(2*n, n) for n in range(1, 4)]
ph = {m: np.exp(-2j*np.pi*(m[0]*I + m[1]*J)/L) for m in modes}
comp = dict(E34=lambda P, E: E[0][..., 3, 4] + E[1][..., 3, 4] + E[2][..., 3, 4], P01=lambda P, E: P[..., 0, 1], P34=lambda P, E: P[..., 3, 4], P03=lambda P, E: P[..., 0, 3], P00=lambda P, E: P[..., 0, 0] - P[..., 1, 1])
rec = {k: [] for k in comp}; t = 0.0
while t < T - 1e-9:
    P, Pi, U, E = g.step(P, Pi, U, E, dt); t += dt
    if abs(t*4 - round(t*4)) < dt/2:
        for k, f in comp.items(): x = f(P, E); rec[k].append([np.sum(x*ph[m]) for m in modes])
np.savez(f'disp_ka{KA:g}_kb{KB:g}.npz', modes=np.array(modes), dts=0.25, L=L, **{k: np.array(v) for k, v in rec.items()})
print('done')
