"""run316 = run315 + зерно (gr=R): правильне кругле зерно в центрі скла.
run315: аморфна Матриця (тріангуляція Делоне rnd51, зсув вузлів jit) з теперішніми нитками бази (β 1,3, стріли з ізотропного τ ℓ_τ 16,
запас 3,5, n 100) і пружністю P 0,16 (ℓ 8, γ 0,0025), переписаною на граф: e = (L_граф + 1/ℓ²)⁻¹ u (на правильній ґратці = FFT ela139).
Без Φ★ і калібрувального поля (pf=0): на випадковій сітці їх ще нема. Ключі: jit= (0 — правильна ґратка), gseed=, seed=, Tt=, ck=."""
import sys, os, json, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl
sys.path.insert(0, os.environ.get("CODE", "/mnt/project-files/model/s52/r268/code"))
import ela139 as E, local60 as M, rnd51 as R
GR = 0.0; JIN = 0.3; TB = 0   # (316): кругле зерно радіуса GR (зсув JIN) у центрі скла зі зсувом jit; TB 1 — τ у незсунутих точках (гладкі стріли, змінюється лише граф)
_orig = R.RandFabric.__init__
def _init(self, L, gseed=0, jit=1.0, rep=1):
    if not GR and not TB: return _orig(self, L, gseed, jit, rep)
    rng = np.random.default_rng(1000 + gseed); n = L*L
    i, j = np.divmod(np.arange(n), L); X = i + j/2; Y = j*np.sqrt(3)/2
    P = np.c_[X % L, Y*L/(L*np.sqrt(3)/2)]; u = rng.uniform(-0.5, 0.5, (n, 2))
    r = np.hypot(*(((P - L/2) + L/2) % L - L/2).T); P0 = P.copy(); P = (P + np.where((r < GR)[:, None], JIN, jit)*u) % L
    self.__dict__.clear(); _build(self, L, P)
    if TB: self.pos_j = self.pos; self.pos = P0
def _build(self, L, P):
    from scipy.spatial import Delaunay
    n = L*L; self.L = L; self.n = n
    img = np.concatenate([P + L*np.array([dx, dy]) for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
    tri = Delaunay(img).simplices; c = img[tri].mean(1)
    keep = (c[:, 0] >= 0) & (c[:, 0] < L) & (c[:, 1] >= 0) & (c[:, 1] < L); tri = tri[keep]; pts = img[tri]
    cr = (pts[:, 1, 0]-pts[:, 0, 0])*(pts[:, 2, 1]-pts[:, 0, 1]) - (pts[:, 1, 1]-pts[:, 0, 1])*(pts[:, 2, 0]-pts[:, 0, 0])
    tri = np.where(cr[:, None] > 0, tri, tri[:, [0, 2, 1]]) % n
    tail, head, eid = [], [], {}
    def edge(u, v):
        key = (min(u, v), max(u, v))
        if key not in eid: eid[key] = len(tail); tail.append(u); head.append(v)
        e = eid[key]; return e, (1 if tail[e] == u else -1)
    Tl = np.array([[*edge(p, q), *edge(q, r), *edge(r, p)] for p, q, r in tri])
    self.tail, self.head = np.array(tail), np.array(head); self.m = len(tail)
    self.E1, self.S1, self.E2, self.S2, self.E3, self.S3 = Tl.T; self.ntri = len(Tl); self.TV = tri; self.pos = P
    self.xy = np.floor(P).astype(int); self.deg = np.bincount(self.tail, None, n) + np.bincount(self.head, None, n)
    assert self.m == 3*n and self.ntri == 2*n
R.RandFabric.__init__ = _init
E.GAM = 0.0025; E.ELL = 8.0
def initR(s, L=24, gseed=0, jit=1.0, h=0.3, lc=1.3, **k):
    R.Sys.__init__(s, L=L, gseed=gseed, jit=jit, beta=BETA, h=h, lc=lc)
    f = s.fab; n = f.n; A = sp.coo_matrix((np.ones(f.m), (f.tail, f.head)), shape=(n, n)); A = (A + A.T).tocsr()
    Lg = sp.diags(np.asarray(A.sum(1)).ravel()) - A
    s.lu = spl.splu((Lg + sp.identity(n)/E.ELL**2).tocsc()); s.gam = E.GAM; s.ell = E.ELL
    r_ = np.roots([1, 0, -1, -h]); s.psiM = r_[np.isreal(r_)].real.min()
M.LocR.__init__ = initR
base = M.Loc.F
def F(s, z, psi, parts=False):
    if not hasattr(s, "lu"): return base(s, z, psi, parts)
    out = E.base_F(s, z, psi, parts)
    if parts: return out
    e = s.lu.solve(psi - s.psiM); return out[0], out[1] - s.gam*e
M.LocR.F = F
BETA = 1.3; kw = dict(L=96, n=100, s0=3.5, ell=16, iso=1, beta=1.3, Tt=300.0, seed=1)
for a in sys.argv[1:]:
    k, v = a.split("=")
    if k == "jit": kw["rnd"] = float(v)
    elif k == "gr": GR = float(v)
    elif k == "gin": JIN = float(v)   # (316б): зсув усередині зерна
    elif k == "tb": TB = int(v)
    else: kw[k] = v if k == "ckdir" else (int(v) if k in ("L", "n", "seed", "gseed", "iso", "ck") else float(v))
if kw.get("rnd", 0) == 0: E.GAM = 0.0025; E.ELL = 8.0   # правильна ґратка: ela139 як у базі (FFT)
r = M.run(**kw)
f = None
json.dump(r, open(os.path.basename(r["name"]) + (f"_gr{GR:g}" if GR else "") + (f"_gin{JIN:g}" if GR and JIN != 0.3 else "") + ("_tb" if TB else "") + "_r315.json", "w"))
print("done", r["name"], flush=True)
