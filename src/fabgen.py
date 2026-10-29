"""fabgen (114): загальна 3D-сітка трикутників на кубічному торі L³ — для перевірки гіпотези «у 3D трикутників більше, ніж ниток».
Нитки — типи u ∈ U (id = t·n + вузол). Трикутники — трійки (x, x+u, x+v), u, v, v−u ∈ ±U, з фільтром:
  'all'    — усі (FCC: 8 трикутників і 6 ниток на вузол, 8:6);
  'planar' — лише в координатних площинах (ті самі 6 ниток, 6 трикутників, 6:6);
  'layer'  — нитки e1, e2, e3, e1−e2, e1−e3, трикутники в площинах xy і xz (5 ниток, 4 трикутники, 4:5; у 2D 3:2 → 2:3);
  'tet'    — (117) тетраедрична сітка Куна: нитки e1, e2, e3, e1+e2, e2+e3, e1+e3, e1+e2+e3, кожен кубик = 6 тетраедрів, усі клітинки — тетраедри (7 ниток, 12 трикутників на вузол).
Місцевий закон P = I − (D − A)/(1 + d_max) (подвійно стохастична при будь-якій кількості сусідів нитки; на правильній 2D-ґратці — те саме, що (I + A)/5)."""
import numpy as np, scipy.sparse as sp
import simNlib as N
LATT = {"all": [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, -1, 0), (0, 1, -1), (1, 0, -1)],
        "planar": [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, -1, 0), (0, 1, -1), (1, 0, -1)],
        "layer": [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, -1, 0), (1, 0, -1)],
        "tet": [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]}
class FabricG(N.Fabric):
    def __init__(s, L, kind="layer"):
        s.L = L; n = L**3; s.n = n; s.kind = kind; U = np.array(LATT[kind]); s.U = U; nt = len(U); s.nt = nt
        I = np.indices((L, L, L)).reshape(3, -1).T; s.ijk = I
        idx = lambda c: ((c[..., 0] % L)*L + (c[..., 1] % L))*L + (c[..., 2] % L)
        s.tail = np.concatenate([np.arange(n)]*nt); s.head = np.concatenate([idx(I + u) for u in U]); s.m = nt*n
        nbv = np.concatenate([U, -U]); key = {tuple(v) for v in nbv}
        def link(x, u):
            for t, w in enumerate(U):
                if tuple(u) == tuple(w): return t*n + x, np.ones_like(x)
                if tuple(u) == tuple(-w): return t*n + idx(I[x] + u), -np.ones_like(x)
        def keep(u, v):
            if kind in ("all", "tet"): return True
            M = np.array([u, v]); zero = (M == 0).all(0)                 # координата, що дорівнює 0 в обох векторах → площина
            if not zero.any(): return False
            if kind == "planar": return True
            return zero[2] or zero[1]                                     # layer: площини xy (z = 0) і xz (y = 0)
        tri = set(); TV = []; T = []; x = np.arange(n)
        for a, u in enumerate(nbv):
            for b, v in enumerate(nbv):
                if a < b and tuple(v - u) in key and keep(u, v):
                    q = idx(I + u); r = idx(I + v); e1, s1 = link(x, u); e2, s2 = link(q, v - u); e3, s3 = link(r, -v)
                    for j in range(n):
                        k_ = tuple(sorted((j, q[j], r[j])))
                        if k_ in tri: continue
                        tri.add(k_); TV.append((j, q[j], r[j])); T.append((e1[j], s1[j], e2[j], s2[j], e3[j], s3[j]))
        T = np.array(T); s.E1, s.S1, s.E2, s.S2, s.E3, s.S3 = T.T; s.TV = np.array(TV); s.ntri = len(T)
        s.ktri = np.bincount(np.r_[s.E1, s.E2, s.E3], None, s.m)           # трикутників на нитку
        s.tnode = np.bincount(s.TV.ravel(), None, n)                        # трикутників на вузол
        s.deg = np.bincount(s.tail, None, n) + np.bincount(s.head, None, n)
        s.xy = I; s.pos = I.astype(float)
    def wilson(s, z): return (0.0, 0.0)
    def disp(s): return np.repeat(s.U.astype(float), s.n, axis=0)

def local_P(f):
    E3 = np.stack([f.E1, f.E2, f.E3], 1); rows = []; cols = []
    for a, b in ((0, 1), (1, 2), (0, 2)): rows += [E3[:, a], E3[:, b]]; cols += [E3[:, b], E3[:, a]]
    A = sp.csr_matrix((np.ones(sum(len(r) for r in rows)), (np.concatenate(rows), np.concatenate(cols))), shape=(f.m, f.m)); A.data[:] = 1.0
    d = np.asarray(A.sum(1)).ravel(); P = (sp.identity(f.m) - (sp.diags(d) - A)/(1.0 + d.max())).tocsr()
    assert np.allclose(np.asarray(P.sum(0)).ravel(), 1.0) and np.allclose(np.asarray(P.sum(1)).ravel(), 1.0); return P

def fft_K(f, P, n):
    L = f.L; nn = f.n; nt = f.nt; W = np.zeros((nt, nt, L, L, L)); Pc = P.tocsr()
    for t in range(nt):
        row = Pc.getrow(t*nn)
        for c, w in zip(row.indices, row.data):
            t2, y = divmod(int(c), nn); W[t, t2][np.unravel_index(y, (L, L, L))] += w
    Pk = np.conj(np.fft.fftn(W, axes=(2, 3, 4))).transpose(2, 3, 4, 0, 1); Kk = np.linalg.matrix_power(Pk, n)
    def K(src):
        S = np.fft.fftn(src.reshape(nt, L, L, L), axes=(1, 2, 3)).transpose(1, 2, 3, 0)
        return np.real(np.fft.ifftn(np.einsum("...ab,...b->...a", Kk, S).transpose(3, 0, 1, 2), axes=(1, 2, 3))).reshape(-1)
    return K

def tau_cubic(L, ell, seed):
    rng = np.random.default_rng(700 + seed); w = rng.standard_normal((L, L, L)); k = np.fft.fftfreq(L)*2*np.pi
    K2 = k[:, None, None]**2 + k[None, :, None]**2 + k[None, None, :]**2
    t = np.real(np.fft.ifftn(np.fft.fftn(w)*np.exp(-K2*ell**2/2))); return t.ravel()/t.std()
