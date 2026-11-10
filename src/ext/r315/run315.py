"""run315: аморфна Матриця (тріангуляція Делоне rnd51, зсув вузлів jit) з теперішніми нитками бази (β 1,3, стріли з ізотропного τ ℓ_τ 16,
запас 3,5, n 100) і пружністю P 0,16 (ℓ 8, γ 0,0025), переписаною на граф: e = (L_граф + 1/ℓ²)⁻¹ u (на правильній ґратці = FFT ela139).
Без Φ★ і калібрувального поля (pf=0): на випадковій сітці їх ще нема. Ключі: jit= (0 — правильна ґратка), gseed=, seed=, Tt=, ck=."""
import sys, os, json, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl
sys.path.insert(0, os.environ.get("CODE", "/mnt/project-files/model/s52/r268/code"))
import ela139 as E, local60 as M, rnd51 as R
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
    else: kw[k] = v if k == "ckdir" else (int(v) if k in ("L", "n", "seed", "gseed", "iso", "ck") else float(v))
if kw.get("rnd", 0) == 0: E.GAM = 0.0025; E.ELL = 8.0   # правильна ґратка: ela139 як у базі (FFT)
r = M.run(**kw)
f = None
json.dump(r, open(os.path.basename(r["name"]) + "_r315.json", "w"))
print("done", r["name"], flush=True)
