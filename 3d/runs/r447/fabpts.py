"""fabpts (20.11, (447)): 3D-сітка ниток з довільних точок на періодичному ящику — для «світу на ґратці, вирощеній з поля».
Правило сусідів (замість ручного вибору 14 напрямків Куна): два вузли з'єднані, якщо їхні комірки Вороного мають спільну грань
(= ребро тріангуляції Делоне). Для ОЦК комірка — зрізаний октаедр (8+6 граней) → рівно 14 сусідів, як у Куна.
Петлі: 3 чотирикутники (гамільтонові цикли) кожного тетраедра Делоне — як у rq351.quads для Куна.
Трикутники (для місцевого закону і діагностики) — грані тетраедрів.
FabricP(tets, n, pos, box): спільний конструктор; fromPoints(P, box) — через Делоне; fromKun(L) — тетраедри Куна (контроль)."""
import numpy as np, itertools, sys
sys.path.insert(0, "/mnt/project-files/model/s52"); import simNlib as N
from scipy.spatial import Delaunay

class FabricP(N.Fabric):
    def __init__(s, tets, n, pos, box):
        s.n = n; s.nt = 0; s.pos = pos; s.box = float(box); s.tets = tets; s.L = int(round(n**(1/3)))
        e = np.sort(np.concatenate([tets[:, [i, j]] for i, j in itertools.combinations(range(4), 2)]), 1)
        e = np.unique(e, axis=0); s.tail, s.head = e[:, 0].copy(), e[:, 1].copy(); s.m = len(e)
        key = s.tail.astype(np.int64)*n + s.head; o = np.argsort(key); ks = key[o]
        def eid(u, v):
            a = np.minimum(u, v).astype(np.int64); b = np.maximum(u, v); p = np.searchsorted(ks, a*n + b); assert (ks[p] == a*n + b).all()
            ed = o[p]; return ed, np.where(s.tail[ed] == u, 1, -1)
        tri = np.unique(np.sort(np.concatenate([tets[:, list(c)] for c in itertools.combinations(range(4), 3)]), 1), axis=0)
        s.TV = tri; s.ntri = len(tri)
        (s.E1, s.S1), (s.E2, s.S2), (s.E3, s.S3) = eid(tri[:, 0], tri[:, 1]), eid(tri[:, 1], tri[:, 2]), eid(tri[:, 2], tri[:, 0])
        s.ktri = np.bincount(np.r_[s.E1, s.E2, s.E3], None, s.m); s.tnode = np.bincount(tri.ravel(), None, n)
        s.deg = np.bincount(s.tail, None, n) + np.bincount(s.head, None, n)
        s.ijk = np.floor(pos*s.L/s.box).astype(int); s.xy = s.ijk
    def quads(s):
        a, b, c, d = s.tets.T
        return np.concatenate([np.stack(v, 1) for v in ((a, b, c, d), (a, b, d, c), (a, c, b, d))])
    def wilson(s, z): return (0.0, 0.0)
    def disp(s):
        d = s.pos[s.head] - s.pos[s.tail]; return d - s.box*np.round(d/s.box)

def fromKun(L):
    I = np.indices((L, L, L)).reshape(3, -1).T; idx = lambda c: ((c[..., 0] % L)*L + (c[..., 1] % L))*L + (c[..., 2] % L); T = []
    for p in itertools.permutations(range(3)):
        e0 = np.zeros(3, int); e0[p[0]] = 1; e1 = e0.copy(); e1[p[1]] = 1
        T.append(np.stack([np.arange(L**3), idx(I + e0), idx(I + e1), idx(I + 1)], 1))
    return FabricP(np.concatenate(T), L**3, I.astype(float), L)

def fromPoints(P, box):
    """Періодична Делоне: 27 образів, беремо тетраедри, у яких найменший (за образом) вузол — у центральній копії; дублікати знімаємо за відсортованими вершинами."""
    n = len(P); sh = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
    img = np.concatenate([P + box*v for v in sh]); im = np.repeat(np.arange(27), n); c0 = int(np.flatnonzero((sh == 0).all(1))[0])
    D = Delaunay(img); S = D.simplices
    cen = img[S].mean(1); keep = ((cen >= 0) & (cen < box)).all(1)        # кожен тетраедр тора рівно раз: центроїд у центральній клітині
    T = S[keep] % n; T = T[np.lexsort(np.sort(T, 1).T[::-1])]
    st = np.sort(T, 1); u, ix = np.unique(st, axis=0, return_index=True)
    if len(u) != len(T): print("fabpts: дублікати тетраедрів", len(T) - len(u), flush=True)
    T = T[np.sort(ix)]
    assert (np.sort(T, 1)[:, :-1] != np.sort(T, 1)[:, 1:]).all(), "виродження: тетраедр з повтором вузла (ящик замалий?)"
    return FabricP(T, n, P, box)

def tau_points(f, ell, seed):
    """Те саме τ, що fabgen.tau_cubic(L, ell, seed), на сітці L³ ящика; у довільних точках — трилінійна інтерполяція (на Куні точно)."""
    import fabgen as G
    L = f.L; t = G.tau_cubic(L, ell, seed).reshape(L, L, L); x = f.pos*L/f.box; i0 = np.floor(x).astype(int); w = x - i0; out = 0
    for c in itertools.product((0, 1), repeat=3):
        c = np.array(c); wt = np.prod(np.where(c, w, 1 - w), 1); j = (i0 + c) % L; out = out + wt*t[j[:, 0], j[:, 1], j[:, 2]]
    return out
