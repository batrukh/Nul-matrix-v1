"""kund (22.11, (463)): сітка Куна в d вимірах (d = 3 — та сама, що fabgen 'tet'; d = 4 — нова).
Нитки — усі непорожні підмножини e1..ed (2^d − 1 типів, z = 2^(d+1) − 2 нитки на вузол).
Симплекси — ланцюги x, x+e_p1, x+e_p1+e_p2, … (d! на вузол). Петлі — як у 3D: 4-цикли (3 на тетраедр)
на кожній тетраедричній грані симплексу (C(d+1, 4) граней, спільні грані без повторів). У 3D це рівно rq351f.quads.
Трикутники (діагностика й окіл місцевого закону) — як fabgen 'all'/'tet': (x, x+u, x+v), u, v, v−u ∈ ±U.
Також: tau_d, fft_K_d (з Kk_half для fastpatch), krfft_d. Встановлення: install(D) — після fastpatch.install()."""
import itertools, numpy as np
import sys; sys.path[:0] = ["/mnt/project-files/model/s52/r351", "/mnt/project-files/model/s52"]
import simNlib as N

def subsets(d):
    return [tuple(int(i in c) for i in range(d)) for k in range(1, d + 1) for c in itertools.combinations(range(d), k)]

class FabricD(N.Fabric):
    def __init__(s, L, d, U=None):
        s.L = L; s.d = d; n = L**d; s.n = n; s.kind = f"kun{d}"
        U = np.array(U if U is not None else subsets(d)); s.U = U; nt = len(U); s.nt = nt
        I = np.indices((L,)*d).reshape(d, -1).T; s.ijk = I
        s.idx = idx = lambda c: np.ravel_multi_index(tuple((c % L).T), (L,)*d) if c.ndim == 2 else np.ravel_multi_index(tuple(c % L), (L,)*d)
        s.tail = np.concatenate([np.arange(n)]*nt); s.head = np.concatenate([idx(I + u) for u in U]); s.m = nt*n
        tid = {tuple(u): t for t, u in enumerate(U)}
        nbv = np.concatenate([U, -U]); key = {tuple(v) for v in nbv}; x = np.arange(n)
        def link(xx, u):
            u = tuple(u)
            if u in tid: return tid[u]*n + xx, np.ones_like(xx)
            t = tid[tuple(-np.array(u))]; return t*n + idx(I[xx] + np.array(u)), -np.ones_like(xx)
        TV = []; T = []; seen = set()
        for a, u in enumerate(nbv):
            for b, v in enumerate(nbv):
                if a < b and tuple(v - u) in key:
                    q = idx(I + u); r = idx(I + v); e1, s1 = link(x, u); e2, s2 = link(q, v - u); e3, s3 = link(r, -v)
                    TV.append(np.stack([x, q, r], 1)); T.append(np.stack([e1, s1, e2, s2, e3, s3], 1))
        TV = np.concatenate(TV); T = np.concatenate(T)
        _, first = np.unique(np.sort(TV, 1), axis=0, return_index=True); first = np.sort(first)
        TV = TV[first]; T = T[first]
        s.E1, s.S1, s.E2, s.S2, s.E3, s.S3 = T.T; s.TV = TV; s.ntri = len(T)
        s.ktri = np.bincount(np.r_[s.E1, s.E2, s.E3], None, s.m); s.tnode = np.bincount(TV.ravel(), None, n)
        s.deg = np.bincount(s.tail, None, n) + np.bincount(s.head, None, n)
        s.xy = I; s.pos = I.astype(float)
    def wilson(s, z): return (0.0, 0.0)
    def disp(s):   # лише для члена нахилу g_ (у прогонах g_ = 0): перші 3 координати
        return np.repeat(s.U[:, :3].astype(float), s.n, axis=0)

def quads_d(f):
    d = f.d; I = f.ijk; n = f.n; Q = []
    for p in itertools.permutations(range(d)):
        chain = [np.arange(n)]; e = np.zeros(d, int)
        for k in p: e = e.copy(); e[k] = 1; chain.append(f.idx(I + e))
        for c4 in itertools.combinations(range(d + 1), 4):
            a, b, c, dd = (chain[j] for j in c4)
            Q.append(np.stack([np.stack(v, 1) for v in ((a, b, c, dd), (a, b, dd, c), (a, c, b, dd))], 0))   # (3, n, 4)
    Q = np.concatenate(Q, 1)                                                     # (3, ntet_raw, 4)
    tet = np.sort(Q[0], 1); _, first = np.unique(tet, axis=0, return_index=True); first = np.sort(first)
    return np.concatenate([Q[0][first], Q[1][first], Q[2][first]])

def tau_d(L, ell, seed, d):
    rng = np.random.default_rng(700 + seed); w = rng.standard_normal((L,)*d); k = np.fft.fftfreq(L)*2*np.pi
    K2 = sum(np.meshgrid(*([k**2]*d), indexing="ij"))
    t = np.real(np.fft.ifftn(np.fft.fftn(w)*np.exp(-K2*ell**2/2))); return t.ravel()/t.std()

def fft_K_d(f, P, n):
    L = f.L; d = f.d; nn = f.n; nt = f.nt; sh = (L,)*d; ax = tuple(range(2, 2 + d)); W = np.zeros((nt, nt) + sh); Pc = P.tocsr()
    for t in range(nt):
        row = Pc.getrow(t*nn)
        for c, w in zip(row.indices, row.data):
            t2, y = divmod(int(c), nn); W[t, t2][np.unravel_index(y, sh)] += w
    Pk = np.conj(np.fft.fftn(W, axes=ax)).transpose(*ax, 0, 1); Kk = np.linalg.matrix_power(Pk, n)
    axs = tuple(range(1, 1 + d))
    def K(src):
        S = np.moveaxis(np.fft.fftn(src.reshape((nt,) + sh), axes=axs), 0, -1)
        return np.real(np.fft.ifftn(np.moveaxis(np.einsum("...ab,...b->...a", Kk, S), -1, 0), axes=axs)).reshape(-1)
    K.Kk_half = np.ascontiguousarray(Kk[(slice(None),)*(d - 1) + (slice(0, L//2 + 1),)]); return K

def krfft_d(s, src):
    f = s.fab; L = f.L; d = getattr(f, "d", 3); nt = f.nt; sh = (L,)*d; axs = tuple(range(1, 1 + d))
    S = np.moveaxis(np.fft.rfftn(src.reshape((nt,) + sh), axes=axs), 0, -1)
    out = np.moveaxis(np.einsum("...ab,...b->...a", s._Kk, S), -1, 0)
    return np.fft.irfftn(out, s=sh, axes=axs).reshape(-1)

def install(D, U=None):
    import fabgen as G, fastpatch as FP, rq351f as Q
    G.FabricG = lambda L, kind="tet": FabricD(L, D, U)
    G.tau_cubic = lambda L, ell, seed: tau_d(L, ell, seed, D)
    G.fft_K = fft_K_d; FP._krfft = krfft_d; Q.quads = quads_d
