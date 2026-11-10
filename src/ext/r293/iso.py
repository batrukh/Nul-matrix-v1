"""(293) принагідно: чи однакова швидкість «фотона» в різних напрямках. Вакуум 3+2 (a −1, b 0,1), L48, шум 1e-3, без тертя, T од.
Фур'є-моди E34 (сума трьох ланок) з індексами (n,0) — напрям −30° (між сусідами), і (2n,n) — 0° (уздовж ланки).
python3 iso.py [T]"""
import sys, numpy as np
sys.path.insert(0, '/mnt/project-files/model/s52/r268/code')
import gauge91 as G, phi90 as PH
T = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
L, dt, b = 48, 0.05, 0.1
g = G.Gauge(-np.ones((L, L)), np.ones((3, L, L)), np.ones((2, L, L)), 1.0, 1.0, b)
P0 = np.diag(np.sort(np.linalg.eigvalsh(PH.vacuum(-1.0, b))))
P = np.broadcast_to(P0, (L, L, 5, 5)).copy(); U = np.broadcast_to(np.eye(5), (3, L, L, 5, 5)).copy()
rng = np.random.default_rng(11); A = 1e-3
Pi = PH.sym0(A*rng.standard_normal(P.shape)); E = A*rng.standard_normal(U.shape); E = 0.5*(E - np.swapaxes(E, -1, -2))
I, J = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
modes = [(n, 0) for n in range(1, 7)] + [(2*n, n) for n in range(1, 4)] + [(0, n) for n in range(1, 7)]
ph = {m: np.exp(-2j*np.pi*(m[0]*I + m[1]*J)/L) for m in modes}
rec = []; t = 0.0
while t < T - 1e-9:
    P, Pi, U, E = g.step(P, Pi, U, E, dt); t += dt
    if abs(t*4 - round(t*4)) < dt/2:
        f = E[0][..., 3, 4] + E[1][..., 3, 4] + E[2][..., 3, 4]; rec.append([np.sum(f*ph[m]) for m in modes])
np.savez('iso.npz', modes=np.array(modes), rec=np.array(rec), dts=0.25, L=L)
print('done')
