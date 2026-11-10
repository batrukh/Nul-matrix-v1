"""(320) Обмін вихорів у повній моделі: паралельне перенесення зв'язаних мод сектору заряду G01 = +1 (комплексні, J = R01(π/2) → +i)
уздовж φ = π·k/NS, k = 0…NS; кінець: стани при π порівнюються з G·(стани при 0), G = diag(1,1,1,−1,−1).
Одночастинкові фази, двобозонні фази «по кванту на вихорі» (перманенти, як r313), основний стан A_L A_R. python3 braid.py NS [KB WCUT LAST]"""
import numpy as np, sys, json, itertools
from scipy.linalg import expm
sys.path.insert(0, '/mnt/project-files/model/s52/r320'); from pair import *
NS = int(sys.argv[1]); KB = int(sys.argv[2]) if len(sys.argv) > 2 else 8; WCUT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.265
LAST = sys.argv[4] if len(sys.argv) > 4 else 'G0'          # G0 — кінець через G·V0; k — через обчислені моди при π
D = '/mnt/project-files/model/s52/r320'; L = 48; R = 8.0
def tag(k):
    if k % (NS//24) == 0: return f'L48_R8_A0.5_w2_N24_k{k//(NS//24)}'      # спільні точки з NS 24
    return f'L48_R8_A0.5_w2_N{NS}_k{k}'
S = []
for i in range(5):
    for j in range(i, 5):
        M = np.zeros((5, 5)); M[i, j] = M[j, i] = 1; S.append(PH.sym0(M))
Q_ = []
for M in S:
    for q in Q_: M = M - np.sum(M*q)*q
    if np.linalg.norm(M) > 1e-9: Q_.append(M/np.linalg.norm(M))
S = np.array(Q_); T = []
for i in range(5):
    for j in range(i + 1, 5):
        M = np.zeros((5, 5)); M[i, j] = 1/np.sqrt(2); M[j, i] = -1/np.sqrt(2); T.append(M)
T = np.array(T)
def amb_op(Ra):                                            # дія сталого O(5)-перетворення на амбієнтні координати вузла (49)
    B = np.zeros((49, 49)); B[:14, :14] = np.einsum('aij,bkl,ik,jl->ab', S, S, Ra, Ra); B[14:19, 14:19] = Ra
    Mt = np.einsum('aij,bkl,ik,jl->ab', T, T, Ra, Ra)
    for m in range(3): B[19 + 10*m:29 + 10*m, 19 + 10*m:29 + 10*m] = Mt
    return B
Qg = np.zeros((5, 5)); Qg[0, 1], Qg[1, 0] = -1, 1
J = amb_op(expm(np.pi/2*Qg)); Gb = amb_op(G5)
app = lambda B, V: np.einsum('ab,xybk->xyak', B, V.reshape(L, L, 49, -1)).reshape(-1, V.shape[-1])
def load(k):
    z = np.load(f'{D}/modes/{tag(k)}.npz'); w = np.sqrt(np.abs(z['w2'])); nb = int((w < WCUT).sum())
    V = z['Va'].reshape(-1, z['Va'].shape[-1])[:, :KB]; Jr = V.T @ app(J, V); e, u = np.linalg.eig(Jr)
    sel = np.where(np.imag(e) > 0.5)[0]; C = V @ u[:, sel]; C, _ = np.linalg.qr(C)            # сектор +i
    CV = C.conj().T @ V; Hc = CV @ np.diag(z['w2'][:KB]) @ CV.conj().T
    return C, w, nb, Hc
import heapq
Zb = amb_op(Z5)
def cfg(k): z = np.load(f'{D}/fin/{tag(k)}.npz'); return z['U'], z['P'], z['n']
nsign = lambda g, na, nb: np.sign(np.einsum('xyi,xyij,xyj->xy', nb, g, na) + 1e-300)
def align(Ua, Ub):
    """g(x) ∈ {I, Z}: Ub(x,m) ≈ g(x)·Ua(x,m)·g(x+d)ᵀ (розріз пересунувся). Жадібне дерево від далекого вузла."""
    Sx = pos(L); far = np.unravel_index(np.argmax(wrapv(Sx - centre(L), L)[0]), (L, L))
    g = -np.ones((L, L), int); g[far] = 0; Gs = [np.eye(5), Z5]; h = []; worst = 0.0
    def push(x):
        for m, d in enumerate(DIRS):
            y = ((x[0] + d[0]) % L, (x[1] + d[1]) % L)
            if g[y] < 0:
                e = [np.abs(Ub[m][x] - Gs[g[x]] @ Ua[m][x] @ Gs[s].T).max() for s in (0, 1)]; heapq.heappush(h, (min(e), y, int(np.argmin(e))))
            y = ((x[0] - d[0]) % L, (x[1] - d[1]) % L)
            if g[y] < 0:
                e = [np.abs(Ub[m][y] - Gs[s] @ Ua[m][y] @ Gs[g[x]].T).max() for s in (0, 1)]; heapq.heappush(h, (min(e), y, int(np.argmin(e))))
    push(far)
    while h:
        e, y, s = heapq.heappop(h)
        if g[y] >= 0: continue
        g[y] = s; worst = max(worst, e); push(y)
    return g, worst
def refine(g01, Ua, Ub, Pa, Pb, sweeps=60, lam=1.0):
    """уточнення калібрування g(x) ∈ O(5) (неперервний поворот біля рухомих ядер): максимум Σ tr(U_bᵀ g U_a g'ᵀ) + λ tr(P_b g P_a gᵀ),
    почергово по вузлах (полярний розклад), старт — розв'язок {I, Z} з дерева."""
    g = np.where((g01 == 1)[..., None, None], Z5, np.eye(5)); T_ = lambda A: np.swapaxes(A, -1, -2)
    sh = lambda A, d: np.roll(A, (-d[0], -d[1]), axis=(0, 1)); put = lambda A, d: np.roll(A, (d[0], d[1]), axis=(0, 1))
    def resid(g): return max(np.abs(Ub[m] - g @ Ua[m] @ T_(sh(g, d))).max() for m, d in enumerate(DIRS))
    r0 = resid(g); par = (np.arange(L)[:, None] + 2*np.arange(L)[None, :]) % 3          # 3-розфарбування трикутної ґратки
    for it in range(sweeps):
        for c in range(3):
            M = lam*(Pa @ T_(g) @ Pb)
            for m, d in enumerate(DIRS):
                M = M + Ua[m] @ T_(sh(g, d)) @ T_(Ub[m]) + put(T_(Ua[m]) @ T_(g) @ Ub[m], d)
            W, s_, Vt = np.linalg.svd(M); gn = T_(Vt) @ T_(W)
            msk = par == c; g[msk] = gn[msk]
    return g, r0, resid(g)
def gapply(g, V, sg=None):
    Vr = V.reshape(L, L, 49, -1); Bs = np.zeros((L, L, 49, 49))
    Bs[..., :14, :14] = np.einsum('aij,bkl,xyik,xyjl->xyab', S, S, g, g); Bs[..., 14:19, 14:19] = g if sg is None else g*sg[..., None, None]          # директор n ≡ −n: знак вузла
    Mt = np.einsum('aij,bkl,xyik,xyjl->xyab', T, T, g, g)
    for m in range(3): Bs[..., 19 + 10*m:29 + 10*m, 19 + 10*m:29 + 10*m] = Mt
    return np.einsum('xyab,xybk->xyak', Bs, Vr).reshape(V.shape)
C0, w0, nb0, H0 = load(0); K = C0.shape[1]; print('k0: зв\'язаних', nb0, 'ω', np.round(w0[:KB + 2], 4), 'комплексних', K)
Vk = C0.copy(); smin = []; nbs = [nb0]; wmax = [w0[KB - 1]]; wthr = [w0[KB]]
ks = list(range(1, NS + (1 if LAST == 'k' else 0)))
Uprev, Pprev, nprev = cfg(0); gerr = []; nflip = []
for k in ks:
    Cn, w, nb, _ = load(k); Uk, Pk, nk = cfg(k); g01, ge = align(Uprev, Uk); gk, ra, rb = refine(g01, Uprev, Uk, Pprev, Pk); gerr.append(float(rb))
    sg = nsign(gk, nprev, nk); nflip.append(int((sg < 0).sum())); Vk = gapply(gk, Vk, sg); Uprev, Pprev, nprev = Uk, Pk, nk
    M = Vk.conj().T @ Cn; u, s, vh = np.linalg.svd(M); smin.append(float(s.min()))
    Vk = Cn @ (vh.conj().T @ u.conj().T); nbs.append(nb); wmax.append(w[KB - 1]); wthr.append(w[KB])
if LAST == 'G0':
    Cn = app(Gb, C0); M = Vk.conj().T @ Cn; u, s, vh = np.linalg.svd(M); smin.append(float(s.min())); Vk = Cn @ (vh.conj().T @ u.conj().T)
if LAST == 'k':                                     # конфігурація при π проти G·(конфігурація при 0)·G з точністю до розрізу
    U0_, P0_, n0_ = cfg(0); UG = np.einsum('ij,mxyjk,kl->mxyil', G5, U0_, G5); PG = G5 @ P0_ @ G5
    g01, geF = align(UG, Uprev); gF, ra, rb = refine(g01, UG, Uprev, PG, Pprev); gerr.append(float(rb)); sgF = nsign(gF, n0_ @ G5, nprev); nflip.append(int((sgF < 0).sum())); Uh = (gapply(gF, app(Gb, C0), sgF)).conj().T @ Vk
else: Uh = C0.conj().T @ app(Gb, Vk)                      # стан при π, повернутий назад G (G⁻¹ = G), у базисі станів при 0
Sx = pos(L); c = centre(L); X = np.repeat(wrapv(Sx - c, L)[1][..., 0][..., None], 49, -1).reshape(-1)
Xm = C0.conj().T @ (X[:, None]*C0); xq, Q = np.linalg.eigh(Xm); Ul = Q.conj().T @ Uh @ Q; Hl = Q.conj().T @ H0 @ Q
H2 = K//2; fl = np.zeros((K, K), complex); lev = []
for b in (0, H2):
    e_, f_ = np.linalg.eigh(Hl[b:b + H2, b:b + H2]); fl[b:b + H2, b:b + H2] = f_; lev.append(np.sqrt(np.abs(e_)).round(4).tolist())
Ul = fl.conj().T @ Ul @ fl
def perm(A): return sum(np.prod([A[i, s[i]] for i in range(A.shape[0])]) for s in itertools.permutations(range(A.shape[0])))
st = list(itertools.combinations_with_replacement(range(K), 2))
def amp(p, q):
    A = Ul[np.ix_(p, q)]; return perm(A)/np.sqrt((2 if p[0] == p[1] else 1)*(2 if q[0] == q[1] else 1))
S2 = np.array([[amp(p, q) for q in st] for p in st])
one = [i for i, p in enumerate(st) if p[0] < H2 <= p[1]]; same = [i for i, p in enumerate(st) if i not in one]
e1 = np.linalg.eigvals(Uh); eo = np.linalg.eigvals(S2[np.ix_(one, one)])
gg = amp((0, H2), (0, H2))                                  # основний рівень на кожному вихорі: A_L A_R
res = dict(nflip=nflip, gauge_err=float(max(gerr)) if gerr else None, gerrs=np.round(gerr, 4).tolist(), NS=NS, LAST=LAST, K=K, nb=nbs, smin=float(min(smin)), smins=np.round(smin, 4).tolist(), wbound_max=float(max(wmax)), wthr_min=float(min(wthr)),
           xq=np.round(xq, 2).tolist(), levels=lev, unit_err=float(np.abs(Uh.conj().T @ Uh - np.eye(K)).max()),
           phases1=sorted((np.angle(e1)/np.pi).round(4).tolist()), two_one_per_vortex=sorted((np.angle(eo)/np.pi).round(4).tolist()),
           AA_phase=float(np.angle(gg)/np.pi), AA_abs=float(abs(gg)), Ul_abs=np.round(np.abs(Ul), 3).tolist(), Ul_ph=np.round(np.angle(Ul)/np.pi, 3).tolist(),
           leak=float(np.abs(S2[np.ix_(same, one)]).max()))
json.dump(res, open(f'{D}/braid_N{NS}_{LAST}.json', 'w')); print(json.dumps({k: res[k] for k in res if k not in ('Ul_abs', 'Ul_ph')}, ensure_ascii=False))
print('|Ul|', np.round(np.abs(Ul), 3)); print('arg Ul/π', np.round(np.angle(Ul)/np.pi, 3))
