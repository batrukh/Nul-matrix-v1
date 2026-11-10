"""(313) Обмін двох вихорів, кожен із зв'язаним непарним квантом (ефективна модель eff.py).
Паралельне перенесення 4 зв'язаних мод уздовж φ 0 → PM·π; голономія в локалізованому базисі;
двобозонні фази (перманенти, S²U) для «по кванту на вихорі» і «обидва на одному».
python3 braid.py L R NSTEP PM MODE TAG   (MODE odd|even)"""
import numpy as np, sys, json, time, itertools
from eff import *
L, R, NS, PM, MODE, TAG = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5], sys.argv[6]
import os; PAR = dict(t=0.1, rc=float(os.environ.get("RC", 4.0)), h=float(os.environ.get("H", 1.0)), minf=0.333, m3=0.6); K = int(os.environ.get("K", 4)); H2 = K//2
c = np.array([0.75*L + 0.25, np.sqrt(3)/4*L + 0.2])
def modes(phi):
    n, U, r, vv = config(L, R, 3.0, phi, c=c); D = build(L, n, U, r, mode=MODE, **PAR)
    w, v = low(D, K + 4); return w, v, D
t0 = time.time(); phis = np.linspace(0, PM*np.pi, NS + 1)
w, V0, D0 = modes(0.0); w00 = w.copy(); gap = float(np.sqrt(w[K]) - np.sqrt(w[K - 1])); Vk = V0[:, :K].copy(); wmax = [np.sqrt(w[:K]).max()]; wthr = [np.sqrt(w[K])]; smin = []
for ph in phis[1:]:
    w, v, _ = modes(ph); Vn = v[:, :K]; M = Vk.T @ Vn
    u, s, vt = np.linalg.svd(M); smin.append(float(s.min())); Vk = Vn @ (vt.T @ u.T)
    wmax.append(np.sqrt(w[:K]).max()); wthr.append(np.sqrt(w[K]))
G = np.tile([1.0, -1.0], L*L) if (MODE == 'odd' and abs(PM - 1) < 1e-9) else np.ones(2*L*L)
_, _, D1 = modes(PM*np.pi); symerr = float(abs(D1 - sp.diags(G) @ D0 @ sp.diags(G)).max())
Uh = V0[:, :K].T @ (G[:, None]*Vk)
x, y = np.meshgrid(np.arange(L), np.arange(L), indexing='ij'); X = np.repeat((x + 0.5*y).ravel(), 2)
Xm = V0[:, :K].T @ (X[:, None]*V0[:, :K]); xq, Q = np.linalg.eigh(Xm); Ul = Q.T @ Uh @ Q    # 0,1 — лівий вихор, 2,3 — правий
def perm(A):
    return sum(np.prod([A[i, s[i]] for i in range(A.shape[0])]) for s in itertools.permutations(range(A.shape[0])))
st = list(itertools.combinations_with_replacement(range(K), 2))
def amp(p, q):
    A = Ul[np.ix_(p, q)]; nrm = np.sqrt((2 if p[0] == p[1] else 1)*(2 if q[0] == q[1] else 1)); return perm(A)/nrm
if K == 4:                                             # «аромати»: власні стани енергії в межах дублета кожного вихору
    Hq = Q.T @ np.diag(w00[:K]) @ Q; fl = np.zeros((4, 4))
    for b in (0, 2): e_, f_ = np.linalg.eigh(Hq[b:b+2, b:b+2]); fl[b:b+2, b:b+2] = f_
    Ul = fl.T @ Ul @ fl; split = float(np.sqrt(w00[3]) - np.sqrt(w00[0]))
S2 = np.array([[amp(p, q) for q in st] for p in st])
one = [k for k, p in enumerate(st) if p[0] < H2 <= p[1]]; same = [k for k, p in enumerate(st) if k not in one]
e1 = np.linalg.eigvals(Uh); eo = np.linalg.eigvals(S2[np.ix_(one, one)]); es = np.linalg.eigvals(S2[np.ix_(same, same)])
res = dict(L=L, R=R, NS=NS, PM=PM, MODE=MODE, gap_w=gap, wbound_max=float(max(wmax)), wthr_min=float(min(wthr)), smin=min(smin), symerr=symerr,
           xq=np.round(xq, 2).tolist(), Ul=np.round(Ul, 3).tolist(),
           phases1=sorted((np.angle(e1)/np.pi).round(4).tolist()),
           two_one_per_vortex=sorted((np.angle(eo)/np.pi).round(4).tolist()), two_same=sorted((np.angle(es)/np.pi).round(4).tolist()),
           flav_same=[float(np.angle(amp((0, 2), (0, 2)))/np.pi) if K == 4 else None, float(np.angle(amp((1, 3), (1, 3)))/np.pi) if K == 4 else None], flav_same_abs=[float(abs(amp((0, 2), (0, 2)))) if K == 4 else None, float(abs(amp((1, 3), (1, 3)))) if K == 4 else None], leak=float(np.abs(S2[np.ix_(same, one)]).max()), sec=round(time.time() - t0))
json.dump(res, open(f'braid_{TAG}.json', 'w')); print(json.dumps({k: res[k] for k in res if k != 'Ul'}, ensure_ascii=False)); print(np.round(Ul, 3))
