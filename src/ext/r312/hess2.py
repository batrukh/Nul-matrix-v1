"""(312) як hess.py (306), але з директором n (4 дотичні координати на вузол, маса 1) зі стану fin_{TAG}.npz.
python3 hess2.py RR TAG [VACTAG]
(306) Оператор малих коливань (Φ★ + калібрувальне SO(5)) навколо вихору бази r294 (κa 1, κb 2, b 0,1), область r < RR від v1.
Координати: P += Σ p_k S_k (14 ортонорм. симетричних безслідових), U → exp(Σ a_k T_k)·U (10 ортонорм. антисиметричних).
Маси: p — 1, a — κa/2. Якобіан сил скінченними різницями з розфарбуванням (крок 6). python3 hess.py RR"""
import numpy as np, sys, time, scipy.sparse as sp
sys.path.insert(0, '/mnt/project-files/model/s52/r268/code'); import gauge91 as G, phi90 as PH
from scipy.linalg import expm
RR = float(sys.argv[1]); TAG = sys.argv[2]; KA, KB, b = 1.0, 2.0, 0.1
sys.path.insert(0, '/tmp/claude-0/-home-user/11ca481a-2b4a-53b5-a097-dab19d222dd0/scratchpad/r312'); from gd import GD
z = np.load('/mnt/project-files/model/s52/r294/e0_ka1_kb2_a-1_L96.npz'); L = 96
zf = np.load(f'/mnt/project-files/model/s52/r312/fin_{TAG}.npz'); MU = float(TAG.replace('mu', '').replace('00', '0.0', 1)) if False else None
cx, cy = 50.5, 25.1147
X, Y = PH.coords(L); r = PH.tdist(X, Y, cx, cy, L)
j0 = int(round(cy/(np.sqrt(3)/2))); i0 = int(round(cx - 0.5*j0))
WI, WJ = 2*int(RR + 0.5*RR + 6), 2*int(RR/0.866 + 6)
ii = (i0 - WI//2 + np.arange(WI)) % L; jj = (j0 - WJ//2 + np.arange(WJ)) % L
P0 = zf['P'][np.ix_(ii, jj)].copy(); U0 = zf['U'][:, ii][:, :, jj].copy(); n0 = zf['n'][np.ix_(ii, jj)].copy(); rw = r[np.ix_(ii, jj)]
import os
MU = float(os.environ['MU']); KD = float(os.environ.get('KD', 0.09))
if len(sys.argv) > 3:                                                   # контроль: релаксований вакуум із директором (fin_vac*)
    zv = np.load(f'/mnt/project-files/model/s52/r312/fin_{sys.argv[3]}.npz')
    P0 = np.broadcast_to(zv['P'][0, 0], P0.shape).copy(); U0 = np.broadcast_to(np.eye(5), U0.shape).copy(); n0 = np.broadcast_to(zv['n'][0, 0], n0.shape).copy()
g = GD(-np.ones((WI, WJ)), np.ones((3, WI, WJ)), np.ones((2, WI, WJ)), KA, KB, b, mu=MU, k=KD)
# дотичний базис директора
BT = np.zeros((WI, WJ, 4, 5))
for a_ in range(WI):
    for b_ in range(WJ):
        q_, _ = np.linalg.qr(np.column_stack([n0[a_, b_], np.eye(5)])); BT[a_, b_] = q_[:, 1:5].T
reg = rw < RR
# базиси
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
DIRS = G.DIRS
# індекси ступенів свободи
pidx = -np.ones((WI, WJ), int); n = 0
sites = np.argwhere(reg)
for (a_, b_) in sites: pidx[a_, b_] = n; n += 1
NP = n; aidx = -np.ones((3, WI, WJ), int); n = 0
for m, d in enumerate(DIRS):
    ok = reg & np.roll(reg, (-d[0], -d[1]), axis=(0, 1))
    for (a_, b_) in np.argwhere(ok): aidx[m, a_, b_] = n; n += 1
NA = n; ND = 4*NP; N = 14*NP + ND + 10*NA
print('sites', NP, 'links', NA, 'dof', N, flush=True)
def gforce(P, U, nn=None):
    nn = n0 if nn is None else nn
    FP, FE, Fn = g.forces_d(P, U, nn)
    fp = np.einsum('kij,xyij->xyk', S, FP); fa = (KA/2)*np.einsum('kij,mxyij->mxyk', T, FE); fn = np.einsum('xyki,xyi->xyk', BT, Fn)
    return fp, fa, fn
fp0, fa0, fn0 = gforce(P0, U0)
_, _, Fn0 = g.forces_d(P0, U0, n0); lam0 = np.sum(Fn0*n0, -1)          # f·n0 для поправки сфери
print('залишкова сила в області: P', np.abs(fp0[reg]).max(), 'n', np.abs(fn0[reg]).max(), 'A', max(np.abs(fa0[m][aidx[m] >= 0]).max() for m in range(3)), flush=True)
eps = 1e-4; rows, cols, vals = [], [], []
def vec(fp, fa, fn):
    v = np.zeros(N); v[:14*NP] = fp[reg].ravel(); v[14*NP:14*NP + ND] = fn[reg].ravel()
    for m in range(3):
        msk = aidx[m] >= 0; v[14*NP + ND + 10*aidx[m][msk][:, None] + np.arange(10)] = fa[m][msk]
    return v
# для кожного рядка: базовий вузол (i, j)
rowsite = np.zeros((N, 2), int); rowsite[:14*NP] = np.repeat(sites, 14, 0); rowsite[14*NP:14*NP + ND] = np.repeat(sites, 4, 0)
for m in range(3):
    for (a_, b_) in np.argwhere(aidx[m] >= 0): rowsite[14*NP + ND + 10*aidx[m, a_, b_] + np.arange(10)] = (a_, b_)
t0 = time.time(); SP = 6
for ci in range(SP):
    for cj in range(SP):
        cm = np.zeros((WI, WJ), bool); cm[ci::SP, cj::SP] = True
        dl = ((rowsite[:, 0] - ci + 3) % SP) - 3; dm = ((rowsite[:, 1] - cj + 3) % SP) - 3
        okr = (np.abs(dl) <= 2) & (np.abs(dm) <= 2); src = rowsite - np.stack([dl, dm], 1)
        srcin = okr & (src[:, 0] >= 0) & (src[:, 0] < WI) & (src[:, 1] >= 0) & (src[:, 1] < WJ)
        for typ in range(14 + 4 + 30):
            if typ < 14:
                msk = cm & reg
                if not msk.any(): continue
                dP = np.zeros_like(P0); dP[msk] = S[typ]
                fpp, fap, fnp = gforce(P0 + eps*dP, U0); fpm, fam, fnm = gforce(P0 - eps*dP, U0)
                colof = lambda s: np.where(pidx[s[:, 0], s[:, 1]] >= 0, 14*pidx[s[:, 0], s[:, 1]] + typ, -1)
            elif typ < 18:
                kk = typ - 14; msk = cm & reg
                np_ = n0.copy(); nm_ = n0.copy(); np_[msk] += eps*BT[msk][:, kk]; nm_[msk] -= eps*BT[msk][:, kk]
                np_ /= np.linalg.norm(np_, axis=-1, keepdims=True); nm_ /= np.linalg.norm(nm_, axis=-1, keepdims=True)
                fpp, fap, fnp = gforce(P0, U0, np_); fpm, fam, fnm = gforce(P0, U0, nm_)
                colof = lambda s, kk=kk: np.where(pidx[s[:, 0], s[:, 1]] >= 0, 14*NP + 4*pidx[s[:, 0], s[:, 1]] + kk, -1)
            else:
                m, k = divmod(typ - 18, 10); msk = cm & (aidx[m] >= 0)
                if not msk.any(): continue
                Rp = np.broadcast_to(np.eye(5), U0.shape).copy(); Rm = Rp.copy()
                Rp[m][msk] = expm(eps*T[k]); Rm[m][msk] = expm(-eps*T[k])
                fpp, fap, fnp = gforce(P0, Rp @ U0); fpm, fam, fnm = gforce(P0, Rm @ U0)
                colof = lambda s, m=m, k=k: np.where(aidx[m][s[:, 0], s[:, 1]] >= 0, 14*NP + ND + 10*aidx[m][s[:, 0], s[:, 1]] + k, -1)
            dv = -(vec(fpp, fap, fnp) - vec(fpm, fam, fnm))/(2*eps)
            c = -np.ones(N, int); s_ = np.clip(src, 0, [WI - 1, WJ - 1]); c[srcin] = colof(s_[srcin])
            nz = (c >= 0) & (np.abs(dv) > 1e-9)
            rows.append(np.nonzero(nz)[0]); cols.append(c[nz]); vals.append(dv[nz])
        print('колір', ci, cj, round(time.time() - t0), flush=True)
K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
K = K + sp.diags(np.concatenate([np.zeros(14*NP), np.repeat(lam0[reg], 4), np.zeros(10*NA)]))   # поправка сфери: +f·n0
asym = abs(K - K.T).max(); print('асиметрія', asym, 'max', abs(K).max(), flush=True)
K = 0.5*(K + K.T)
mass = np.concatenate([np.ones(14*NP + ND), (KA/2)*np.ones(10*NA)])
# калібрувальні напрямки: X_n (10) у вузлі n: δP_n = [X,P_n]; δA_m(n) = X_n − U X_{n+d} Uᵀ
br, bc, bv = [], [], []; col = 0
for (a_, b_) in sites:
    for k in range(10):
        X_ = T[k]; dp = X_ @ P0[a_, b_] - P0[a_, b_] @ X_
        e = 14*pidx[a_, b_] + np.arange(14); br += list(e); bc += [col]*14; bv += list(np.einsum('kij,ij->k', S, dp))
        e = 14*NP + 4*pidx[a_, b_] + np.arange(4); br += list(e); bc += [col]*4; bv += list(BT[a_, b_] @ (X_ @ n0[a_, b_]))
        for m, d in enumerate(DIRS):
            if aidx[m, a_, b_] >= 0:
                e = 14*NP + ND + 10*aidx[m, a_, b_] + np.arange(10); br += list(e); bc += [col]*10; bv += list(np.einsum('kij,ij->k', T, X_))
            q = ((a_ - d[0]) % WI, (b_ - d[1]) % WJ)
            if aidx[m][q] >= 0:
                Uq = U0[m][q]; e = 14*NP + ND + 10*aidx[m][q] + np.arange(10); br += list(e); bc += [col]*10; bv += list(np.einsum('kij,ij->k', T, -Uq @ X_ @ Uq.T))
        col += 1
B = sp.csr_matrix((bv, (br, bc)), shape=(N, col))
gres = np.abs(K @ B).max(); print('K·B (калібрувальна інваріантність)', gres, flush=True)
sp.save_npz('K.npz', K); sp.save_npz('B.npz', B)
np.savez('meta.npz', n0=n0, BT=BT, ND=ND, mass=mass, sites=sites, NP=NP, NA=NA, aidx=aidx, pidx=pidx, ii=ii, jj=jj, rw=rw, P0=P0, U0=U0, S=S, T=T, cx=cx, cy=cy, i0=i0, j0=j0, WI=WI, WJ=WJ)
print('done', round(time.time() - t0))
