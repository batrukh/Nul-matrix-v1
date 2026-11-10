"""(306) Оператор малих коливань (Φ★ + калібрувальне SO(5)) навколо вихору бази r294 (κa 1, κb 2, b 0,1), область r < RR від v1.
Координати: P += Σ p_k S_k (14 ортонорм. симетричних безслідових), U → exp(Σ a_k T_k)·U (10 ортонорм. антисиметричних).
Маси: p — 1, a — κa/2. Якобіан сил скінченними різницями з розфарбуванням (крок 6). python3 hess.py RR"""
import numpy as np, sys, time, scipy.sparse as sp
sys.path.insert(0, '/mnt/project-files/model/s52/r268/code'); import gauge91 as G, phi90 as PH
from scipy.linalg import expm
RR = float(sys.argv[1]); KA, KB, b = 1.0, 2.0, 0.1
z = np.load('/mnt/project-files/model/s52/r294/e0_ka1_kb2_a-1_L96.npz'); L = 96
cx, cy = 50.5, 25.1147
X, Y = PH.coords(L); r = PH.tdist(X, Y, cx, cy, L)
j0 = int(round(cy/(np.sqrt(3)/2))); i0 = int(round(cx - 0.5*j0))
WI, WJ = 2*int(RR + 0.5*RR + 6), 2*int(RR/0.866 + 6)
ii = (i0 - WI//2 + np.arange(WI)) % L; jj = (j0 - WJ//2 + np.arange(WJ)) % L
P0 = z['P'][np.ix_(ii, jj)].copy(); U0 = z['U'][:, ii][:, :, jj].copy(); rw = r[np.ix_(ii, jj)]
import os
if os.environ.get('VAC'):                                               # контроль: однорідний вакуум бази в тій самій області
    Pv = np.diag(np.sort(np.linalg.eigvalsh(PH.vacuum(-1.0, b)))); P0 = np.broadcast_to(Pv, P0.shape).copy(); U0 = np.broadcast_to(np.eye(5), U0.shape).copy()
g = G.Gauge(-np.ones((WI, WJ)), np.ones((3, WI, WJ)), np.ones((2, WI, WJ)), KA, KB, b)
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
NA = n; N = 14*NP + 10*NA
print('sites', NP, 'links', NA, 'dof', N, flush=True)
def gforce(P, U):
    FP, FE = g.forces(P, U)
    fp = np.einsum('kij,xyij->xyk', S, FP); fa = (KA/2)*np.einsum('kij,mxyij->mxyk', T, FE)
    return fp, fa
fp0, fa0 = gforce(P0, U0)
print('залишкова сила в області: P', np.abs(fp0[reg]).max(), 'A', max(np.abs(fa0[m][aidx[m] >= 0]).max() for m in range(3)), flush=True)
eps = 1e-4; rows, cols, vals = [], [], []
def vec(fp, fa):
    v = np.zeros(N); v[:14*NP] = fp[reg].ravel()
    for m in range(3):
        msk = aidx[m] >= 0; v[14*NP + 10*aidx[m][msk][:, None] + np.arange(10)] = fa[m][msk]
    return v
# для кожного рядка: базовий вузол (i, j)
rowsite = np.zeros((N, 2), int); rowsite[:14*NP] = np.repeat(sites, 14, 0)
for m in range(3):
    for (a_, b_) in np.argwhere(aidx[m] >= 0): rowsite[14*NP + 10*aidx[m, a_, b_] + np.arange(10)] = (a_, b_)
t0 = time.time(); SP = 6
for ci in range(SP):
    for cj in range(SP):
        cm = np.zeros((WI, WJ), bool); cm[ci::SP, cj::SP] = True
        dl = ((rowsite[:, 0] - ci + 3) % SP) - 3; dm = ((rowsite[:, 1] - cj + 3) % SP) - 3
        okr = (np.abs(dl) <= 2) & (np.abs(dm) <= 2); src = rowsite - np.stack([dl, dm], 1)
        srcin = okr & (src[:, 0] >= 0) & (src[:, 0] < WI) & (src[:, 1] >= 0) & (src[:, 1] < WJ)
        for typ in range(14 + 30):
            if typ < 14:
                msk = cm & reg
                if not msk.any(): continue
                dP = np.zeros_like(P0); dP[msk] = S[typ]
                fpp, fap = gforce(P0 + eps*dP, U0); fpm, fam = gforce(P0 - eps*dP, U0)
                colof = lambda s: np.where(pidx[s[:, 0], s[:, 1]] >= 0, 14*pidx[s[:, 0], s[:, 1]] + typ, -1)
            else:
                m, k = divmod(typ - 14, 10); msk = cm & (aidx[m] >= 0)
                if not msk.any(): continue
                Rp = np.broadcast_to(np.eye(5), U0.shape).copy(); Rm = Rp.copy()
                Rp[m][msk] = expm(eps*T[k]); Rm[m][msk] = expm(-eps*T[k])
                fpp, fap = gforce(P0, Rp @ U0); fpm, fam = gforce(P0, Rm @ U0)
                colof = lambda s, m=m, k=k: np.where(aidx[m][s[:, 0], s[:, 1]] >= 0, 14*NP + 10*aidx[m][s[:, 0], s[:, 1]] + k, -1)
            dv = -(vec(fpp, fap) - vec(fpm, fam))/(2*eps)
            c = -np.ones(N, int); s_ = np.clip(src, 0, [WI - 1, WJ - 1]); c[srcin] = colof(s_[srcin])
            nz = (c >= 0) & (np.abs(dv) > 1e-9)
            rows.append(np.nonzero(nz)[0]); cols.append(c[nz]); vals.append(dv[nz])
        print('колір', ci, cj, round(time.time() - t0), flush=True)
K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
asym = abs(K - K.T).max(); print('асиметрія', asym, 'max', abs(K).max(), flush=True)
K = 0.5*(K + K.T)
mass = np.concatenate([np.ones(14*NP), (KA/2)*np.ones(10*NA)])
# калібрувальні напрямки: X_n (10) у вузлі n: δP_n = [X,P_n]; δA_m(n) = X_n − U X_{n+d} Uᵀ
br, bc, bv = [], [], []; col = 0
for (a_, b_) in sites:
    for k in range(10):
        X_ = T[k]; dp = X_ @ P0[a_, b_] - P0[a_, b_] @ X_
        e = 14*pidx[a_, b_] + np.arange(14); br += list(e); bc += [col]*14; bv += list(np.einsum('kij,ij->k', S, dp))
        for m, d in enumerate(DIRS):
            if aidx[m, a_, b_] >= 0:
                e = 14*NP + 10*aidx[m, a_, b_] + np.arange(10); br += list(e); bc += [col]*10; bv += list(np.einsum('kij,ij->k', T, X_))
            q = ((a_ - d[0]) % WI, (b_ - d[1]) % WJ)
            if aidx[m][q] >= 0:
                Uq = U0[m][q]; e = 14*NP + 10*aidx[m][q] + np.arange(10); br += list(e); bc += [col]*10; bv += list(np.einsum('kij,ij->k', T, -Uq @ X_ @ Uq.T))
        col += 1
B = sp.csr_matrix((bv, (br, bc)), shape=(N, col))
gres = np.abs(K @ B).max(); print('K·B (калібрувальна інваріантність)', gres, flush=True)
sp.save_npz('K.npz', K); sp.save_npz('B.npz', B)
np.savez('meta.npz', mass=mass, sites=sites, NP=NP, NA=NA, aidx=aidx, pidx=pidx, ii=ii, jj=jj, rw=rw, P0=P0, U0=U0, S=S, T=T, cx=cx, cy=cy, i0=i0, j0=j0, WI=WI, WJ=WJ)
print('done', round(time.time() - t0))
