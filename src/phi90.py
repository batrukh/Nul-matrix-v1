"""phi90: light as massless modes of the protofield Φ★ (author's idea of 21.10.2026: heredity through the Primary field and the protofield, sectorisation;
a check, not a postulate). Φ★ is a real symmetric traceless 5×5 matrix at the nodes of the triangular mesh (as (24), differ13), but with inertia:
    Φ̈ + γΦ̇ = c0²ΔΦ − ∂V/∂Φ + εB,   V = (a/2)trΦ² − (b/3)trΦ³ + (c/4)(trΦ²)² + (d/4)trΦ⁴,
a(x) = a_m + (a_w − a_m)·χ(ψ), χ = (1 + tanh(ψ/0.2))/2: in a domain (ψ > 0) a_w < 0 — Φ★ splits, in the Matrix a_m > 0 — it does not.
ψ is taken from a finished local60 state and frozen (the links z are not touched: α does not act on Φ★).
Δ — triangular mesh, 6 neighbours (±1,0),(0,±1),(1,−1),(−1,1), Δ = (2/3)Σ(Φn − Φ) (→ ∇² at small k)."""
import numpy as np, json, sys

NB = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
I5 = np.eye(5)

def lap(P):
    return (2/3)*(sum(np.roll(P, s, axis=(0, 1)) for s in NB) - 6*P)

def sym0(P):
    P = 0.5*(P + np.swapaxes(P, -1, -2)); return P - np.trace(P, axis1=-2, axis2=-1)[..., None, None]*I5/5

def dV(P, a, b=0.0, c=1.0, d=0.5):
    P2 = P @ P; t2 = np.trace(P2, axis1=-2, axis2=-1)[..., None, None]
    return sym0(a[..., None, None]*P - b*P2 + c*t2*P + d*(P2 @ P))

def vacuum(a, b=0.0, c=1.0, d=0.5, seed=0):
    """minimum of V for a single matrix (gradient descent from several starts)"""
    best = None
    for s in range(20):
        P = sym0(np.random.default_rng(seed + s).standard_normal((5, 5)))
        for _ in range(20000): P = P - 0.02*dV(P[None], np.array([a]), b, c, d)[0]
        P2 = P @ P; E = a/2*np.trace(P2) - b/3*np.trace(P2 @ P) + c/4*np.trace(P2)**2 + d/4*np.trace(P2 @ P2)
        if best is None or E < best[0] - 1e-10: best = (E, P)
    return best[1]

def hess(P0, a, B=None, eps=0.0, b=0.0, c=1.0, d=0.5, h=1e-5):
    """Hessian of −force in the 14-dimensional space of traceless symmetric matrices (orthonormal basis)"""
    basis = []
    for i in range(5):
        for j in range(i, 5):
            M = np.zeros((5, 5)); M[i, j] = M[j, i] = 1; basis.append(M)
    basis = [sym0(M) for M in basis]; Q = []                       # Gram–Schmidt
    for M in basis:
        for q in Q: M = M - np.sum(M*q)*q
        if np.linalg.norm(M) > 1e-9: Q.append(M/np.linalg.norm(M))
    f = lambda P: dV(P[None], np.array([a]), b, c, d)[0] - (eps*B if eps else 0)
    H = np.array([[np.sum(q*(f(P0 + h*p) - f(P0 - h*p))/(2*h)) for p in Q] for q in Q])
    return np.linalg.eigvalsh(0.5*(H + H.T))

def relax(P, a, eps, B, T=60.0, dt=0.05, gam=1.0):
    V = np.zeros_like(P); t = 0
    while t < T:
        V += dt*(lap(P) - dV(P, a) + eps*B - gam*V); P = sym0(P + dt*V); t += dt
    return P

def evolve(P, V, a, eps, B, dt, gam):
    """velocity Verlet"""
    A = lap(P) - dV(P, a) + eps*B
    V = V + 0.5*dt*(A - gam*V); P = sym0(P + dt*V)
    A = lap(P) - dV(P, a) + eps*B
    V = (V + 0.5*dt*A)/(1 + 0.5*dt*gam)
    return P, V

def coords(L):
    x, y = np.meshgrid(np.arange(L), np.arange(L), indexing="ij"); return x + 0.5*y, np.sqrt(3)/2*y

def tdist(X, Y, cx, cy, L):
    best = None
    for p in (-1, 0, 1):
        for q in (-1, 0, 1):
            d_ = np.hypot(X - cx + p*L + q*L/2, Y - cy + q*np.sqrt(3)/2*L)
            best = d_ if best is None else np.minimum(best, d_)
    return best

def kick_field(P, r, kind, amp, r0=2.0, gen=(0, 4)):
    th = amp*np.exp(-r**2/(2*r0**2))[..., None, None]
    if kind == "gold":                                   # rotation (broken generator) — Goldstone direction
        G = np.zeros((5, 5)); i, j = gen; G[i, j], G[j, i] = 1, -1
        return P + th*(G @ P - P @ G)
    return P + th*P                                       # "amplitude" (massive) direction

def run(src="", L=48, aw=-1.0, am=1.0, eps=0.0, kind="gold", T=30.0, dt=0.05, gam=0.0, amp=0.01, seed=1, cen=""):
    rng = np.random.default_rng(seed)
    if src:
        psi = np.load(src)["psi"]; L = int(round(np.sqrt(len(psi)))); psi = psi.reshape(L, L)
    else:
        psi = np.ones((L, L)) if aw is not None else None
    chi = 0.5*(1 + np.tanh(psi/0.2)); a = am + (aw - am)*chi
    B = sym0(rng.standard_normal((5, 5))); B /= np.linalg.norm(B)          # direction inherited from the Matrix (ε — strength)
    if src: P = relax(0.3*sym0(rng.standard_normal((L, L, 5, 5))), a, eps, B, T=120.0)
    else: P = relax(np.broadcast_to(vacuum(aw) if aw < 0 else np.zeros((5, 5)), (L, L, 5, 5)).copy(), a, eps, B, T=20.0)
    X, Y = coords(L)
    if cen: cx, cy = [float(v) for v in cen.split(",")]
    else: cx, cy = X.mean(), Y.mean()
    r = tdist(X, Y, cx, cy, L)
    PA, VA = P.copy(), np.zeros_like(P); PB, VB = kick_field(P, r, kind, amp), np.zeros_like(P)
    bins = np.arange(0, L/2 + 1, 1.0); cnt = np.maximum(np.histogram(r, bins)[0], 1); rec = []; t = 0.0
    while t < T - 1e-9:
        PA, VA = evolve(PA, VA, a, eps, B, dt, gam); PB, VB = evolve(PB, VB, a, eps, B, dt, gam); t += dt
        if abs(t*2 - round(t*2)) < dt:
            e = np.sum((PB - PA)**2, axis=(-2, -1)) + np.sum((VB - VA)**2, axis=(-2, -1))
            prof = np.histogram(r, bins, weights=e)[0]/cnt
            inw = psi > 0; hot = e > 1e-3*e.max()
            rec.append(dict(t=round(t, 2), prof=[float(v) for v in prof], tot=float(e.sum()), fin=float(e[inw].sum()/e.sum()),
                            reach_w=float(r[hot & inw].max()) if (hot & inw).any() else 0.0, reach_m=float(r[hot & ~inw].max()) if (hot & ~inw).any() else 0.0))
    # orientation of the laws: the leading eigenvector of the block and whether it is the same in different domains
    ev = np.linalg.eigh(P)[1]
    name = f"phi90_{('src_' + src.split('/')[-1][:-4]) if src else 'uni'}_aw{aw:g}_am{am:g}_e{eps:g}_{kind}_g{gam:g}" + (f"_c{cen}" if cen else "")
    np.savez(name + ".npz", P=P, psi=psi, a=a)
    json.dump(dict(rec=rec, bins=list(bins[:-1])), open(name + ".json", "w"))
    return name, rec

if __name__ == "__main__":
    kw = dict(x.split("=") for x in sys.argv[1:])
    kw = {k: (v if k in ("src", "kind", "cen") else (int(v) if k in ("L", "seed") else float(v))) for k, v in kw.items()}
    name, rec = run(**kw); print(name)
    for q in rec[::4]:
        p = np.array(q["prof"]); front = np.nonzero(p > 1e-3*p.max())[0].max()
        print(q["t"], "front", front, "tot %.3g" % q["tot"], flush=True)
