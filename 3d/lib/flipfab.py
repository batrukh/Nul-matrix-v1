"""flipfab: Матриця без розмірності (крок після (107), ідея автора 23.10). Тріангуляція тора без вкладення: старт — правильна трикутна ґратка L×L,
далі q·m випадкових перекидань ребра (2–2: трикутники (a,b,c), (b,a,d) → (a,d,c), (d,b,c)), заборонено: ступінь < 3 і подвійні ребра.
q = 0 — правильна ґратка (контроль); великі q — «загальна» випадкова тріангуляція: кожна нитка, як і раніше, у 2 трикутниках,
але вузол лежить у 3–20+ трикутниках, а розмірність Гаусдорфа ≈ 4 (динамічні тріангуляції), тобто 2D-геометрії немає.
Координат немає: до-час τ — плавне випадкове поле самого графа, τ = exp(−t·Δ_графа)·шум, t = ℓ²/3 (на правильній ґратці це гауссів фільтр масштабу ℓ)."""
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
import simNlib as N

class FlipFabric(N.Fabric):
    def __init__(s, L, q=0.0, gseed=0):
        s.L = L; n = L*L; s.n = n; rng = np.random.default_rng(3000 + gseed)
        idx = lambda x, y: (x % L)*L + (y % L)
        T = []
        for x in range(L):
            for y in range(L):
                a, b, c, d = idx(x, y), idx(x+1, y), idx(x, y+1), idx(x+1, y+1)
                T += [[a, b, c], [b, d, c]]
        T = np.array(T); et = {}                                              # ребро (min,max) → множина трикутників
        def add(k):
            for i in range(3):
                u, v = T[k, i], T[k, (i+1) % 3]; et.setdefault((min(u, v), max(u, v)), set()).add(k)
        def rem(k):
            for i in range(3):
                u, v = T[k, i], T[k, (i+1) % 3]; et[(min(u, v), max(u, v))].discard(k)
        for k in range(len(T)): add(k)
        deg = np.zeros(n, int)
        for (u, v) in et: deg[u] += 1; deg[v] += 1
        nf = int(q*3*n); done = 0; tries = 0; keys = list(et.keys())
        while done < nf and tries < 50*nf + 100:
            tries += 1; e = keys[rng.integers(len(keys))]
            if e not in et or len(et[e]) != 2: continue
            k1, k2 = et[e]; u, v = e
            if deg[u] <= 3 or deg[v] <= 3: continue
            t1 = list(T[k1]); i = [j for j in range(3) if t1[j] in (u, v) and t1[(j+1) % 3] in (u, v)][0]
            a, b, c = t1[i], t1[(i+1) % 3], t1[(i+2) % 3]                    # (a,b,c), орієнтовано
            d = [w for w in T[k2] if w not in (a, b)][0]
            if (min(c, d), max(c, d)) in et or c == d: continue
            rem(k1); rem(k2); del et[e]
            T[k1] = (a, d, c); T[k2] = (d, b, c); add(k1); add(k2)
            keys.append((min(c, d), max(c, d))); deg[u] -= 1; deg[v] -= 1; deg[c] += 1; deg[d] += 1; done += 1
        s.nflip = done
        tail, head, eid = [], [], {}
        def edge(u, v):
            key = (min(u, v), max(u, v))
            if key not in eid: eid[key] = len(tail); tail.append(u); head.append(v)
            e = eid[key]; return e, (1 if tail[e] == u else -1)
        Tl = np.array([[*edge(p, q_), *edge(q_, r), *edge(r, p)] for p, q_, r in T])
        s.tail, s.head = np.array(tail), np.array(head); s.m = len(tail)
        s.E1, s.S1, s.E2, s.S2, s.E3, s.S3 = Tl.T; s.ntri = len(Tl); s.TV = T
        s.deg = np.bincount(s.tail, None, n) + np.bincount(s.head, None, n)
        cnt = np.bincount(np.r_[s.E1, s.E2, s.E3], None, s.m)
        assert s.m == 3*n and s.ntri == 2*n and (cnt == 2).all(), (s.m, s.ntri, np.unique(cnt))
        s.xy = np.c_[np.arange(n)//L, np.arange(n) % L]                     # лише мітки (геометричного змісту при q > 0 немає)
        s.Ad = sp.csr_matrix((np.ones(2*s.m), (np.r_[s.tail, s.head], np.r_[s.head, s.tail])), shape=(n, n))
    def wilson(s, z): return (0.0, 0.0)
    def lap(s): return sp.diags(s.deg.astype(float)) - s.Ad

def tau_graph(f, ell, seed):
    w = np.random.default_rng(700 + seed).standard_normal(f.n)
    t = expm_multiply(-(ell**2/3.0)*f.lap(), w); return t/t.std()

def bfs_dist(f, src, allow=None):
    """відстані по графу від src (у межах маски allow)"""
    A = f.Ad.tocsr(); dist = np.full(f.n, -1); dist[src] = 0; fr = [src]; r = 0
    while fr:
        r += 1; nx = []
        for u in fr:
            for v in A.indices[A.indptr[u]:A.indptr[u+1]]:
                if dist[v] < 0 and (allow is None or allow[v]): dist[v] = r; nx.append(v)
        fr = nx
    return dist

def ball_dim(f, nsrc=40, rmax=None, seed=0, allow=None):
    """розмірність за ростом кулі N(r) ~ r^d: нахил log N по log r на середніх r"""
    rng = np.random.default_rng(seed); cand = np.flatnonzero(allow) if allow is not None else np.arange(f.n)
    Ns = []
    for s0 in rng.choice(cand, min(nsrc, len(cand)), replace=False):
        d = bfs_dist(f, s0, allow); Ns.append(np.bincount(d[d >= 0]).cumsum())
    R = min(len(x) for x in Ns); M = np.mean([x[:R] for x in Ns], 0); return M
