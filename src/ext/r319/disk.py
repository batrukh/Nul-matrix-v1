"""(319) Світ-диск (ψ заморожене, як у gf104: a = 1 − 2χ(ψ), ваги max(χ, 0,02), κa 1, κb 2, b 0,1) з директором n.
Член директора зважено вагами ланок wl («директор спить у Матриці», як калібрувальне поле, варіант А).
Загасна релаксація до max|сили| < TOL. Контрольні точки ck_{TAG}.npz. python3 disk.py MU TAG [L R]"""
import numpy as np, sys, os, time, json
sys.path.insert(0, '/mnt/project-files/model/s52/r319'); from gd import GD, sh, put, PA, DIRS, cayley
sys.path.insert(0, '/mnt/project-files/model/s52/r268/code'); import phi90 as PH, gauge91 as G
MU, TAG = float(sys.argv[1]), sys.argv[2]; L = int(sys.argv[3]) if len(sys.argv) > 3 else 64; R = float(sys.argv[4]) if len(sys.argv) > 4 else 18.0
KD = float(os.environ.get('KD', 0.09)); TOL = 2e-6; OUT = f'/mnt/project-files/model/s52/r319/ck_{TAG}.npz'
class GDW(GD):
    def en(s, P, U, n):
        e = s.mu*np.einsum('...i,...ij,...j', n, P @ P, n)
        for m, d in enumerate(DIRS):
            f = np.einsum('...i,...i', n, np.einsum('...ij,...j', U[m], sh(n, d))); e = e - s.wl[m]*s.k/2*f**2
        return e
    def forces_d(s, P, U, n):
        FP, FE = s.forces(P, U); Pn = np.einsum('...ij,...j', P, n); Fn = -2*s.mu*np.einsum('...ij,...j', P, Pn)
        FP = FP - s.mu*PH.sym0(np.einsum('...i,...j', Pn, n) + np.einsum('...i,...j', n, Pn))
        for m, d in enumerate(DIRS):
            kk = (s.k*s.wl[m])[..., None]
            X = np.einsum('...ij,...j', U[m], sh(n, d)); f = np.sum(n*X, -1)[..., None]
            Fn += kk*f*X; Fn += put(kk*f*np.einsum('...ji,...j', U[m], n), d)
            R_ = -(kk*f)[..., None]*np.einsum('...i,...j', n, X)
            FE[m] += -(2/(s.ka*s.wl[m]))[..., None, None]*PA(R_)
        return FP, FE, Fn
X, Y = PH.coords(L); r = PH.tdist(X, Y, X.mean(), Y.mean(), L)
psi = np.tanh((R - r)/2.0); chi = 0.5*(1 + np.tanh(psi/0.2)); wl, wt = G.weights(chi, 0.02)
g = GDW(1.0 - 2.0*chi, wl, wt, 1.0, 2.0, 0.1, mu=MU, k=KD)
if os.path.exists(OUT):
    c = np.load(OUT); P, Pi, U, E, n, pn, it0 = (c[k] for k in ['P', 'Pi', 'U', 'E', 'n', 'pn', 'it']); it0 = int(it0); print('resume', it0, flush=True)
else:
    Pv = np.diag(np.sort(np.linalg.eigvalsh(PH.vacuum(-1.0, 0.1))))
    P = chi[..., None, None]*Pv; Pi = np.zeros_like(P); U = np.broadcast_to(np.eye(5), (3, L, L, 5, 5)).copy(); E = np.zeros_like(U)
    rng = np.random.default_rng(5); v = rng.normal(size=5); v[3:] *= 0.3; n = np.broadcast_to(v/np.linalg.norm(v), (L, L, 5)).copy()
    n = n + 0.05*rng.normal(size=n.shape); n /= np.linalg.norm(n, axis=-1, keepdims=True); pn = np.zeros_like(n); it0 = 0
t0 = time.time(); dt, gam = 0.1, 0.5; it = it0
tang = lambda v, n: v - np.sum(v*n, -1, keepdims=True)*n
while True:
    P, Pi, U, E, n, pn = g.step_d(P, Pi, U, E, n, pn, dt, gam); it += 1
    if it % 500 == 0:
        FP, FE, Fn = g.forces_d(P, U, n); fm = max(np.abs(FP).max(), np.abs(FE).max(), np.abs(tang(Fn, n)).max())
        print(it, 'сила %.2e' % fm, round(time.time() - t0), flush=True)
        if it % 2000 == 0 or fm < TOL: np.savez(OUT, P=P, Pi=Pi, U=U, E=E, n=n, pn=pn, it=it)
        if fm < TOL or it >= 60000: break
e = g.edens(P, np.zeros_like(P), U, np.zeros_like(U)) + g.en(P, U, n)
w, V = np.linalg.eigh(P); ne = np.abs(np.einsum('xyij,xyi->xyj', V, n))**2          # |n·власний вектор|²
t2 = np.trace(P @ P, axis1=-2, axis2=-1); bins = np.arange(0, L/2, 1.0); idx = np.digitize(r, bins)
prof = [dict(r=float(bins[i - 1]), t2=float(t2[idx == i].mean()), ev=np.round(w[idx == i].mean(0), 4).tolist(),
             n_uv=float(ne[idx == i][:, 3:].sum(-1).mean()), n_least=float(ne[idx == i][np.arange((idx == i).sum()), np.argmin(np.abs(w[idx == i][:, :3]), -1)].mean()),
             e=float(e[idx == i].mean())) for i in range(1, len(bins) + 1) if (idx == i).any()]
res = dict(MU=MU, L=L, R=R, it=it, E=float(e.sum()), prof=prof, sec=round(time.time() - t0))
json.dump(res, open(f'/mnt/project-files/model/s52/r319/disk_{TAG}.json', 'w'))
for q in prof[::2]: print(q)
print('E', res['E'])
