"""(320) Оператор малих коливань на всій періодичній L×L (Φ 14 + директор 4 дотичні + ланки 30 на вузол; маси 1, 1, κa/2),
як hess2 (312), але без області. Сектор |q01| = 1 (заряд під G01, де живе зв'язаний непарний квант (307)/(312)):
D' = Dm + 4CCᵀ + Λ(1 − P1), P1 = (1/8)Σ_j 2cos α_j R01(α_j). Найнижчі NEV мод → амбієнтне представлення
(14 P, 5 n, 30 ланок на вузол, масово зважене) для перенесення. python3 hess.py TAG [NEV SIG]"""
import numpy as np, sys, os, time, json, scipy.sparse as sp
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh
sys.path.insert(0, '/mnt/project-files/model/s52/r320'); from pair import *
TAG = sys.argv[1]; NEV = int(sys.argv[2]) if len(sys.argv) > 2 else 12; SIG = float(sys.argv[3]) if len(sys.argv) > 3 else 0.03
D = '/mnt/project-files/model/s52/r320'; z = np.load(f'{D}/fin/{TAG}.npz'); P0_, U0, n0 = z['P'], z['U'], z['n']; L = P0_.shape[0]
g = model(L, float(z['R']), float(z['phi']), float(z['A']), float(z['w']))
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
T = np.array(T)
BT = np.zeros((L, L, 4, 5))
for a in range(L):
    for b in range(L):
        q_, _ = np.linalg.qr(np.column_stack([n0[a, b], np.eye(5)])); BT[a, b] = q_[:, 1:5].T
def gf(P, U, n):
    FP, FE, Fn = g.forces_d(P, U, n)
    return np.concatenate([np.einsum('kij,xyij->xyk', S, FP), np.einsum('xyki,xyi->xyk', BT, Fn),
                           np.concatenate([(KA/2)*np.einsum('kij,xyij->xyk', T, FE[m]) for m in range(3)], -1)], -1)
F0 = gf(P0_, U0, n0); lam = np.sum(g.forces_d(P0_, U0, n0)[2]*n0, -1)
print(TAG, 'залишкова сила', np.abs(F0).max(), flush=True)
N = 48*L*L; idx = lambda a, b, t: (a*L + b)*48 + t
eps = 1e-4; SP = 6; rows, cols, vals = [], [], []; t0 = time.time()
A_, B_ = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
for ci in range(SP):
    for cj in range(SP):
        cm = np.zeros((L, L), bool); cm[ci::SP, cj::SP] = True
        dl = ((A_ - ci + 3) % SP) - 3; dm = ((B_ - cj + 3) % SP) - 3; sa = (A_ - dl) % L; sb = (B_ - dm) % L    # джерело для кожного вузла-відгуку
        for t in range(48):
            out = []
            for sg in (1, -1):
                P, U, n = P0_.copy(), U0.copy(), n0.copy()
                if t < 14: P[cm] += sg*eps*S[t]
                elif t < 18:
                    n[cm] += sg*eps*BT[cm][:, t - 14]; n[cm] /= np.linalg.norm(n[cm], axis=-1, keepdims=True)
                else:
                    m, k = divmod(t - 18, 10); U[m][cm] = expm(sg*eps*T[k]) @ U[m][cm]
                out.append(gf(P, U, n))
            dv = -(out[0] - out[1])/(2*eps)                       # (L,L,48)
            ok = (np.abs(dl) <= 2) & (np.abs(dm) <= 2)
            r_ = (((A_*L + B_)*48)[..., None] + np.arange(48))[ok]; c_ = ((sa*L + sb)*48 + t)[ok][:, None] + 0*np.arange(48)
            v_ = dv[ok]; nz = np.abs(v_) > 1e-10
            rows.append(r_[nz]); cols.append(c_[nz]); vals.append(v_[nz])
    print('рядок кольорів', ci, round(time.time() - t0), flush=True)
K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
K = K + sp.diags((np.zeros((L, L, 48)) + np.concatenate([np.zeros(14), np.ones(4), np.zeros(30)])*lam[..., None]).ravel())
print('асиметрія', abs(K - K.T).max(), flush=True); K = 0.5*(K + K.T)
mass = np.tile(np.concatenate([np.ones(18), (KA/2)*np.ones(30)]), L*L)
# калібрувальні напрямки
br, bc, bv = [], [], []
for a in range(L):
    for b in range(L):
        for k in range(10):
            X_ = T[k]; col = (a*L + b)*10 + k
            br += list(idx(a, b, np.arange(14))); bv += list(np.einsum('kij,ij->k', S, X_ @ P0_[a, b] - P0_[a, b] @ X_))
            br += list(idx(a, b, 14 + np.arange(4))); bv += list(BT[a, b] @ (X_ @ n0[a, b]))
            for m, d in enumerate(DIRS):
                br += list(idx(a, b, 18 + 10*m + np.arange(10))); bv += list(np.einsum('kij,ij->k', T, X_))
                q = ((a - d[0]) % L, (b - d[1]) % L); Uq = U0[m][q]
                br += list(idx(q[0], q[1], 18 + 10*m + np.arange(10))); bv += list(np.einsum('kij,ij->k', T, -Uq @ X_ @ Uq.T))
            bc += [col]*(14 + 4 + 60)
Bm = sp.csr_matrix((bv, (br, bc)), shape=(N, 10*L*L)); print('K·B', abs(K @ Bm).max(), flush=True)
# обертання G01
Qg = np.zeros((5, 5)); Qg[0, 1], Qg[1, 0] = -1, 1
def Rblocks(al):
    Ra = expm(al*Qg); Ms = np.einsum('aij,ik,kl,jl->ab', S, Ra, np.eye(5), Ra) if False else np.einsum('aij,bkl,ik,jl->ab', S, S, Ra, Ra)
    Mt = np.einsum('aij,bkl,ik,jl->ab', T, T, Ra, Ra); Mn = np.einsum('xyai,ij,xybj->xyab', BT, Ra, BT)
    blocks = np.zeros((L, L, 48, 48)); blocks[..., :14, :14] = Ms; blocks[..., 14:18, 14:18] = Mn
    for m in range(3): blocks[..., 18 + 10*m:28 + 10*m, 18 + 10*m:28 + 10*m] = Mt
    return blocks
P1b = sum(2*np.cos(al)*Rblocks(al) for al in 2*np.pi*np.arange(8)/8)/8
print('n ⊥ e0,e1:', np.abs(n0[..., :2]).max(), 'P1²−P1', np.abs(np.einsum('xyab,xybc->xyac', P1b, P1b) - P1b).max(), flush=True)
P1 = sp.block_diag(list(P1b.reshape(-1, 48, 48)), format='csr')
Mi = sp.diags(1/np.sqrt(mass)); C = sp.diags(np.sqrt(mass)) @ Bm
LAM = 10.0; Dq = (Mi @ K @ Mi + 4.0*(C @ C.T) + LAM*(sp.identity(N) - P1)).tocsc()
t1 = time.time(); w2, V = eigsh(Dq, k=NEV, sigma=SIG, which='LM'); o = np.argsort(w2); w2, V = w2[o], V[:, o]
print('eigsh', round(time.time() - t1), flush=True)
# амбієнтне представлення: (L,L, 14 + 5 + 30)
Vr = V.reshape(L, L, 48, NEV); Va = np.concatenate([Vr[:, :, :14], np.einsum('xyai,xyak->xyik', BT, Vr[:, :, 14:18]), Vr[:, :, 18:]], 2)
Xs = pos(L); wsite = np.sum(V.reshape(L, L, 48, NEV)**2, 2)
v1, v2 = verts(L, float(z['R']), float(z['phi']))
loc = [dict(w=float(np.sign(a)*np.sqrt(abs(a))), q1=float(V[:, i] @ (P1 @ V[:, i])), core=float(np.sum(wsite[..., i]*((wrapv(Xs - v1, L)[0] < 4) | (wrapv(Xs - v2, L)[0] < 4)))),
            dirw=float(np.sum(Vr[:, :, 14:18, i]**2))) for i, a in enumerate(w2)]
for q in loc: print(f"ω {q['w']:.4f} q1 {q['q1']:.3f} ядро {q['core']:.2f} дир {q['dirw']:.2f}")
os.makedirs(f'{D}/modes', exist_ok=True)
np.savez(f'{D}/modes/{TAG}.npz', w2=w2, Va=Va.astype(np.float64), loc=json.dumps(loc), sec=time.time() - t0)
print('done', round(time.time() - t0), flush=True)
