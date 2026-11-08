"""rnd51: модель stall43 на випадковій трикутній сітці (черга № 40, розд. 4.4 (51)). Перевірка постулату «трикутна комірка»:
чи кипіння, народження і скінченні світи тримаються на трикутниках узагалі, а не на правильній ґратці.
Сітка: N = L² точок на торі [0, L)², розкладених випадково (jit = 1: рівномірно по всьому тору, «пуассонівська»;
jit < 1: правильна трикутна ґратка, кожна точка зсунута випадково на jit·крок). Зв'язки — тріангуляція Делоне на торі
(3×3 копії, беремо трикутники з центроїдом у центральній клітині). Середня кількість сусідів = 6, як у правильній ґратці
(формула Ейлера на торі), тож параметри (β, ε₀, λ_c, α_c) переносяться без перерахунку; різниться лише розкид сусідів (≈ 4–9).
Інтерфейс — як simNlib.Fabric, решта моделі (sim34d.M34, stall43.Sys) без змін."""
import numpy as np, os
from scipy.spatial import Delaunay
os.environ.setdefault("DELTA", "1e-6")
import simNlib as N, sim34d as S, stall43 as T

class RandFabric(N.Fabric):
    def __init__(self, L, gseed=0, jit=1.0, rep=1):
        Ls = L; L = rep*Ls; self.L = L; n = L*L; self.n = n; ns = Ls*Ls; rng = np.random.default_rng(1000 + gseed)
        if rep > 1:                                                  # періодична випадкова сітка: візерунок Ls×Ls, повторений rep×rep
            Q = RandFabric(Ls, gseed, jit).pos
            P = np.concatenate([Q + Ls*np.array([tx, ty]) for tx in range(rep) for ty in range(rep)])
        elif jit >= 1: P = rng.uniform(0, L, (n, 2))
        else:
            i, j = np.divmod(np.arange(n), L); X = i + j/2; Y = j*np.sqrt(3)/2
            P = np.c_[X % L, Y*L/(L*np.sqrt(3)/2)]                 # правильна ґратка, розтягнута на квадрат L×L
            P = (P + jit*rng.uniform(-0.5, 0.5, (n, 2))) % L
        img = np.concatenate([P + L*np.array([dx, dy]) for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
        tri = Delaunay(img).simplices; c = img[tri].mean(1)
        keep = (c[:, 0] >= 0) & (c[:, 0] < L) & (c[:, 1] >= 0) & (c[:, 1] < L); tri = tri[keep]
        pts = img[tri]                                              # орієнтація проти год. стрілки
        cr = (pts[:, 1, 0]-pts[:, 0, 0])*(pts[:, 2, 1]-pts[:, 0, 1]) - (pts[:, 1, 1]-pts[:, 0, 1])*(pts[:, 2, 0]-pts[:, 0, 0])
        tri = np.where(cr[:, None] > 0, tri, tri[:, [0, 2, 1]]) % n
        tail, head, eid = [], [], {}
        def edge(u, v):
            key = (min(u, v), max(u, v))
            if key not in eid: eid[key] = len(tail); tail.append(u); head.append(v)
            e = eid[key]; return e, (1 if tail[e] == u else -1)
        Tl = [[*edge(p, q), *edge(q, r), *edge(r, p)] for p, q, r in tri]
        self.tail, self.head = np.array(tail), np.array(head); self.m = len(tail)
        Tl = np.array(Tl); self.E1, self.S1, self.E2, self.S2, self.E3, self.S3 = Tl.T
        self.ntri = len(Tl); self.TV = tri; self.pos = P
        self.xy = np.floor(P).astype(int)
        deg = np.bincount(self.tail, None, n) + np.bincount(self.head, None, n); self.deg = deg
        cnt = np.bincount(np.r_[self.E1, self.E2, self.E3], None, self.m)
        assert self.m == 3*n and self.ntri == 2*n and (cnt == 2).all(), (self.m, self.ntri, np.unique(cnt))
    def wilson(self, z): return (0.0, 0.0)

class Sys(T.Sys):
    def __init__(s, L=24, gseed=0, jit=1.0, rep=1, beta=1.5, eps0=3.0, h=0.3, kappa=1.0, as_=0.0, ac=1.0, lc=1.3, m0=-1.0):
        s.fab = RandFabric(L, gseed, jit, rep); s.mod = S.M34(s.fab, beta=beta, eps0=eps0, h=h, kappa=kappa)
        s.wp = N.wplus(beta); s.lc, s.m0 = lc, (m0 if m0 > 0 else s.wp**2); s.cs, s.cc = 1 + 1j*as_, 1 + 1j*ac

def flip(fab, mask):
    """Перевернути напрям ниток mask (tail ↔ head). Модель з поворотом сили (α ≠ 0) від напрямку нитки залежить:
    z → z̄ міняє знак обертального члена, тому поле напрямків ниток — фізичне."""
    t = fab.tail.copy(); fab.tail = np.where(mask, fab.head, fab.tail); fab.head = np.where(mask, t, fab.head)
    for k in ("S1", "S2", "S3"):
        E = getattr(fab, "E" + k[1]); setattr(fab, k, np.where(mask[E], -getattr(fab, k), getattr(fab, k)))

def disp(fab):
    """Вектор нитки tail → head (найкоротший образ на торі)."""
    L = fab.L; P = fab.pos if hasattr(fab, "pos") else np.c_[fab.xy[:, 0] + fab.xy[:, 1]/2, fab.xy[:, 1]*np.sqrt(3)/2]
    d = P[fab.head] - P[fab.tail]
    if hasattr(fab, "pos"): d = (d + L/2) % L - L/2
    return d

def orient(fab, mode, rng=None, u=(1.0, 0.31)):
    """mode 'aligned': кожна нитка дивиться «вздовж» сталого вектора u (додатна проекція); 'random': випадкові напрями."""
    if mode == "aligned": mask = disp(fab) @ np.array(u) < 0
    else: mask = rng.random(fab.m) < 0.5
    flip(fab, mask); return float(mask.mean())

class SysO(Sys):
    """Sys на випадковій сітці (або правильній при reg=1) з заданим полем напрямків ниток."""
    def __init__(s, L=24, gseed=0, jit=1.0, rep=1, reg=0, ori="keep", **kw):
        Sys.__init__(s, L=L, gseed=gseed, jit=jit, rep=rep, **kw)
        if reg: s.fab = N.Fabric(L*rep); s.mod = S.M34(s.fab, beta=s.mod.beta, eps0=s.mod.eps0, h=s.mod.h, kappa=s.mod.kappa)
        if ori != "keep": s.flipped = orient(s.fab, ori, np.random.default_rng(500 + gseed))
