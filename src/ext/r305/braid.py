"""(305) Обмін двох вихорів Аліси (+1 у v1, −1 у v2) і голономія Беррі 4 нульових мод ферміона на соті.
Калібрування без розрізів: Φ_n = R(θ_n/2) P0 R(θ_n/2)ᵀ, R — поворот у площині (e2, e3), θ = arg(x−v1) − arg(x−v2);
ланки U = R(−(1−f)δ), δ = wrap(Δθ)/2, f = exp(−(r/r0)²) — калібрувальне поле екранує поворот поза ядром r0.
Обмін = поворот пари на φ: 0 → π навколо центру; кінцевий H = G·H0·G, G = diag(1,1,1,−1,−1) (глобально).
python3 braid.py L R r0 NSTEP PHIMAX(π) TAG"""
import numpy as np, sys, json, time
sys.path.insert(0, '.'); import hc, plant
from scipy.sparse.linalg import eigsh, splu, LinearOperator
L, R, r0, NS, PM, TAG = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
DIRS = plant.DIRS; P0 = plant.P0
x, y = np.meshgrid(np.arange(L), np.arange(L), indexing='ij'); S = np.stack([x + 0.5*y, np.sqrt(3)/2*y], -1)
c = np.array([0.5*L + 0.25*L, 0.5*np.sqrt(3)/2*L])
def config(phi):
    e = np.array([np.cos(phi), np.sin(phi)]); v1, v2 = c + R*e, c - R*e
    th = lambda Q: np.arctan2(Q[..., 1] - v1[1], Q[..., 0] - v1[0]) - np.arctan2(Q[..., 1] - v2[1], Q[..., 0] - v2[0])
    Rn = plant.rot23(th(S)/2); P = Rn @ P0 @ np.swapaxes(Rn, -1, -2)
    U = np.empty((3, L, L, 5, 5))
    for m, d in enumerate(DIRS):
        S2 = S + np.array([d[0] + 0.5*d[1], np.sqrt(3)/2*d[1]]); M = 0.5*(S + S2)
        dth = np.angle(np.exp(1j*(th(S2) - th(S))))
        r = np.minimum(np.hypot(*(M - v1).T).T, np.hypot(*(M - v2).T).T)
        f = np.exp(-(r/r0)**2) if r0 < 1e8 else 1.0 + 0*r
        U[m] = plant.rot23(-(1 - f)*dth/2)
    return P, U, v1, v2
def zero(phi, k=4):
    P, U, v1, v2 = config(phi); H = hc.build(P, U)
    lu = splu(H.tocsc()); Op = LinearOperator(H.shape, matvec=lu.solve, dtype=float)
    w, v = eigsh(Op, k=k + 4, which='LM', ncv=60, tol=1e-11); w = 1/w; o = np.argsort(np.abs(w)); w, v = w[o], v[:, o]
    return w, v, (v1, v2)
t0 = time.time(); phis = np.linspace(0, PM*np.pi, NS + 1)
w, V0, _ = zero(0.0); gap0 = np.abs(w[4:]).min(); Vk = V0[:, :4]; E = [w[:4].tolist()]; smin = []
for ph in phis[1:]:
    w, v, _ = zero(ph); Vn = v[:, :4]
    M = Vk.T @ Vn; sv = np.linalg.svd(M, compute_uv=False); smin.append(float(sv.min()))
    u, s, vt = np.linalg.svd(M); Vk = Vn @ (vt.T @ u.T)            # паралельне перенесення: Vk ← Vn·полярний множник
    E.append(w[:4].tolist())
Gd = np.ones(10*L*L); Gd = Gd.reshape(2, L*L, 5); Gd[..., 3:] = -1; Gd = Gd.ravel()
Gm = Gd if abs(PM - 1) < 1e-9 else np.ones_like(Gd)                # для обміну кінцевий H = G H0 G
Uh = V0[:, :4].T @ (Gm[:, None]*Vk)
XA, YA = hc.coordsAB(L); xs = np.repeat(np.concatenate([XA[:L*L], XA[L*L:]])[:, None], 5, 1).ravel()   # x трикутника для кожної компоненти
Xm = V0[:, :4].T @ (xs[:, None]*V0[:, :4]); xq, Q = np.linalg.eigh(Xm)          # локалізований базис: 2 моди на v1, 2 на v2
ev = np.linalg.eigvals(Uh)
res = dict(Q=Q.tolist(), xq=xq.tolist(), L=L, R=R, r0=r0, NS=NS, PM=PM, gap=float(gap0), Emax=float(np.abs(np.array(E)).max()), smin=min(smin), U=Uh.tolist(),
           phases_over_pi=sorted((np.angle(ev)/np.pi).round(4).tolist()), detU=float(np.linalg.det(Uh)), sec=time.time() - t0)
json.dump(res, open(f'braid_{TAG}.json', 'w')); print(json.dumps({k: res[k] for k in res if k not in ('U', 'Q')})); print(np.round(Uh, 3))
