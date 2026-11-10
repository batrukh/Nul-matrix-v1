"""(313) обмін окремо для кожного «аромату» (рівні 0,1 і 2,3 — розщеплений дублет, по моді на вихорі).
python3 flav.py L R NSTEP PM TAG"""
import numpy as np, sys, json, os, time
from eff import *
L, R, NS, PM, TAG = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
PAR = dict(t=0.1, rc=float(os.environ.get('RC', 4.0)), h=float(os.environ.get('H', 1.0)), minf=0.333, m3=0.6)
c = np.array([0.75*L + 0.25, np.sqrt(3)/4*L + 0.2])
def modes(phi):
    n, U, r, _ = config(L, R, 3.0, phi, c=c); return low(build(L, n, U, r, **PAR), 6)
t0 = time.time(); w, V0 = modes(0.0); Vk = [V0[:, 0:2].copy(), V0[:, 2:4].copy()]; dmin, smin = [], []
for ph in np.linspace(0, PM*np.pi, NS + 1)[1:]:
    w, v = modes(ph); dmin.append(float(np.sqrt(w[2]) - np.sqrt(w[1])))
    for f in range(2):
        Vn = v[:, 2*f:2*f + 2]; u, s, vt = np.linalg.svd(Vk[f].T @ Vn); smin.append(float(s.min())); Vk[f] = Vn @ (vt.T @ u.T)
G = np.tile([1.0, -1.0], L*L) if abs(PM - 1) < 1e-9 else np.ones(2*L*L)
x, y = np.meshgrid(np.arange(L), np.arange(L), indexing='ij'); X = np.repeat((x + 0.5*y).ravel(), 2)
out = dict(L=L, R=R, NS=NS, PM=PM, w0=np.sqrt(w[:6]).round(5).tolist(), dmin=min(dmin), smin=min(smin))
for f in range(2):
    B = V0[:, 2*f:2*f + 2]; Uh = B.T @ (G[:, None]*Vk[f]); xq, Q = np.linalg.eigh(B.T @ (X[:, None]*B)); Ul = Q.T @ Uh @ Q
    perm = Ul[0, 0]*Ul[1, 1] + Ul[0, 1]*Ul[1, 0]
    out[f'flav{f}'] = dict(Ul=np.round(Ul, 4).tolist(), xq=np.round(xq, 2).tolist(), two_boson=[float(abs(perm)), float(np.angle(perm)/np.pi)],
                          phases1=sorted((np.angle(np.linalg.eigvals(Uh))/np.pi).round(4).tolist()))
out['sec'] = round(time.time() - t0); json.dump(out, open(f'flav_{TAG}.json', 'w')); print(json.dumps(out, ensure_ascii=False))
