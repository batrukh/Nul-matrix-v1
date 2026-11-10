"""(319) Дисперсія малих коливань однорідного вакууму Φ★ + калібрувальне SO(5) + директор n (gd.py, база κa 1, κb 2, b 0,1).
Стовпці гессіана для збурень одного вузла (14 P + 4 n + 30 ланок) на періодичній L16; D(k) = Σ_R K(R,0) e^{ik·R};
калібрувальні напрямки B(k) вилучено (фізичний підпростір ⊥ M^{1/2}B). python3 disp.py VACTAG MU KD  (VACTAG=base → вакуум без директора, μ=k=0)"""
import numpy as np, sys, json
sys.path.insert(0, '/mnt/project-files/model/s52/r319'); from gd import GD
sys.path.insert(0, '/mnt/project-files/model/s52/r268/code'); import phi90 as PH, gauge91 as G
from scipy.linalg import expm, null_space
TAG, MU, KD = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]); KA, KB, b = 1.0, 2.0, 0.1; L = 16
if TAG == 'base':
    Pv = PH.vacuum(-1.0, b); nv = np.linalg.eigh(Pv)[1][:, 0]
else:
    zv = np.load(f'/mnt/project-files/model/s52/r312/fin_{TAG}.npz'); Pv = zv['P'][0, 0]; nv = zv['n'][0, 0]
P0 = np.broadcast_to(Pv, (L, L, 5, 5)).copy(); U0 = np.broadcast_to(np.eye(5), (3, L, L, 5, 5)).copy(); n0 = np.broadcast_to(nv, (L, L, 5)).copy()
g = GD(-np.ones((L, L)), np.ones((3, L, L)), np.ones((2, L, L)), KA, KB, b, mu=MU, k=KD)
S = []
for i in range(5):
    for j in range(i, 5):
        M = np.zeros((5, 5)); M[i, j] = M[j, i] = 1; S.append(PH.sym0(M))
Q = []
for M in S:
    for q in Q: M = M - np.sum(M*q)*q
    if np.linalg.norm(M) > 1e-9: Q.append(M/np.linalg.norm(M))
S = np.array(Q); T = []
for i in range(5):
    for j in range(i + 1, 5):
        M = np.zeros((5, 5)); M[i, j] = 1/np.sqrt(2); M[j, i] = -1/np.sqrt(2); T.append(M)
T = np.array(T); BT = np.linalg.qr(np.column_stack([nv, np.eye(5)]))[0][:, 1:5].T
def gf(P, U, n):
    FP, FE, Fn = g.forces_d(P, U, n)
    return np.concatenate([np.einsum('kij,xyij->xyk', S, FP), np.einsum('xyki,xyi->xyk', np.broadcast_to(BT, (L, L, 4, 5)), Fn),
                           np.concatenate([(KA/2)*np.einsum('kij,xyij->xyk', T, FE[m]) for m in range(3)], -1)], -1)   # (L,L,48)
F0 = gf(P0, U0, n0); lam = float(np.sum(g.forces_d(P0, U0, n0)[2][0, 0]*nv)); print('залишкова сила', np.abs(F0).max(), 'λ', lam)
eps = 1e-4; Kc = np.zeros((L, L, 48, 48))
for t in range(48):
    out = []
    for sg in (1, -1):
        P, U, n = P0.copy(), U0.copy(), n0.copy()
        if t < 14: P[0, 0] += sg*eps*S[t]
        elif t < 18: n[0, 0] = (nv + sg*eps*BT[t - 14])/np.linalg.norm(nv + sg*eps*BT[t - 14])
        else: m, k = divmod(t - 18, 10); U[m, 0, 0] = expm(sg*eps*T[k])
        out.append(gf(P, U, n))
    Kc[..., t] = -(out[0] - out[1])/(2*eps)
Kc[0, 0, 14:18, 14:18] += lam*np.eye(4)
mass = np.array([1.0]*18 + [KA/2]*30); Mi = 1/np.sqrt(mass)
i_, j_ = np.meshgrid(np.arange(L), np.arange(L), indexing='ij'); i_ = (i_ + L//2) % L - L//2; j_ = (j_ + L//2) % L - L//2; X = i_ + 0.5*j_; Y = np.sqrt(3)/2*j_   # найближчий образ
def Dk(kx, ky):
    ph = np.exp(-1j*(kx*X + ky*Y)); D = np.einsum('xy,xyab->ab', ph, Kc)     # D_ab = Σ_Δ K(Δ a, 0 b) e^{−ikΔ} (мода v·e^{ikR})
    D = 0.5*(D + D.conj().T); return Mi[:, None]*D*Mi[None, :]
def Bk(kx, ky):                                                             # калібрувальні напрямки при k
    Bm = np.zeros((48, 10), complex)
    for k in range(10):
        Xg = T[k]; Bm[:14, k] = np.einsum('kij,ij->k', S, Xg @ Pv - Pv @ Xg); Bm[14:18, k] = BT @ (Xg @ nv)
        for m, d in enumerate(G.DIRS):
            dx, dy = d[0] + 0.5*d[1], np.sqrt(3)/2*d[1]
            Bm[18 + 10*m + k, k] = 1 - np.exp(1j*(kx*dx + ky*dy))         # X_n − X_{n+d}
    return np.sqrt(mass)[:, None]*Bm
def omega(kx, ky):
    D = Dk(kx, ky); B = Bk(kx, ky); Np = null_space(B.conj().T, rcond=1e-8)
    w2, V = np.linalg.eigh(Np.conj().T @ D @ Np); V = Np @ V
    return w2, V
Gph = np.zeros(10); comp = {}
res = {}
w2, V = omega(0, 0); ev, evec = np.linalg.eigh(Pv)
print('Φ власні', np.round(ev, 4), '|n·власні вектори|²', np.round((evec.T @ nv)**2, 3))
def tag(v):
    wP = np.sum(abs(v[:14])**2); wn = np.sum(abs(v[14:18])**2); wA = np.sum(abs(v[18:])**2); return wP, wn, wA
res['k0'] = [dict(w=float(np.sign(a)*np.sqrt(abs(a))), P=float(tag(V[:, i])[0]), n=float(tag(V[:, i])[1]), A=float(tag(V[:, i])[2])) for i, a in enumerate(w2)]
for q in res['k0']: print(f"ω {q['w']:.4f}  P {q['P']:.2f} n {q['n']:.2f} A {q['A']:.2f}")
ks = np.linspace(0, 0.6, 13); br = []
for kk in ks:
    w2, _ = omega(kk, 0.0); br.append(np.sqrt(np.clip(w2, 0, None)).tolist())
res['kx'] = ks.tolist(); res['branches'] = br; res['min_w2_all'] = float(min(min(omega(a, c)[0]) for a in np.linspace(0, np.pi, 7) for c in np.linspace(0, np.pi, 7)))
print('мін ω² по зоні', res['min_w2_all'])
json.dump(res, open(f'disp_{TAG}.json', 'w'))
