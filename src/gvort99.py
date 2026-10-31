"""gvort99: matter, path A — Z₂ vortices of the protofield with the SO(5) gauge field (gauge91), uniform domain a = −1.
Start: Φ = R(φ)·P0·R(φ)ᵀ, φ = (θ1 − θ2)/2 in the plane (e3, e4); links U_l = R(φ(n) − φ(n+μ)) with the difference wrapped into (−π/2, π/2],
so the covariant derivative ≈ 0 everywhere, and the holonomy R(π) ∈ S(O(3)×O(2)) sits in the two cores (Z₂ flux).
First friction (relaxation to a static state), then free dynamics. We measure: pair energy E(d) (finite mass — E does not grow with d),
cores (triangles with tr H < 1, i.e. holonomy far from 1), motion of the cores."""
import numpy as np, json, sys
import phi90 as PH, gauge91 as G, matter98 as M

def wrap_half(x):
    return (x + np.pi/2) % np.pi - np.pi/2

def init(L, d):
    X, Y = PH.coords(L); cx, cy = X.mean(), Y.mean()
    th = np.arctan2(Y - cy, X - (cx - d/2)) - np.arctan2(Y - cy, X - (cx + d/2)); phi = th/2
    P0 = np.diag([M.U_]*3 + [M.V_]*2); R = M.rot(phi, 2, 3); P = R @ P0 @ np.swapaxes(R, -1, -2)
    U = []
    for dvec in G.DIRS:
        ang = wrap_half(phi - np.roll(phi, (-dvec[0], -dvec[1]), axis=(0, 1)))
        U.append(M.rot(ang, 2, 3))
    rng = np.random.default_rng(1); U = np.array(U)
    for m in range(3):                                   # small noise: the exact lattice configuration is stationary (a saddle)
        Xn = rng.standard_normal(U[m].shape)*0.02; Xn = Xn - np.swapaxes(Xn, -1, -2); U[m] = G.cayley(Xn) @ U[m]
    P = P + PH.sym0(0.01*rng.standard_normal(P.shape))
    return PH.sym0(P), U, X, Y

def cores(g, U, X, Y, L, thr=3.0):
    Hu, Hd = g.hol(U); out = []
    for H, off in ((Hu, (1/3, np.sqrt(3)/6)), (Hd, (2/3, np.sqrt(3)/3))):
        t = np.trace(H, axis1=-2, axis2=-1)
        for k in np.flatnonzero(t < thr):
            i = np.unravel_index(k, t.shape); out.append((float(X[i] + off[0]), float(Y[i] + off[1]), float(t[i])))
    cl = []
    for p in sorted(out, key=lambda q: q[2]):
        if all(min(abs(p[0] - c[0]) % L, L - abs(p[0] - c[0]) % L) > 4 or abs(p[1] - c[1]) > 4 for c in cl): cl.append(p)
    return cl

def wilson(P, U, i0, j0, r, L):
    """holonomy around a parallelogram around (i0, j0) with half-side r; sign of det on the plane v at the base point: −1 — Z₂ flux inside"""
    W = np.eye(5); n = [(i0 - r) % L, (j0 - r) % L]; base = tuple(n)
    for mu, sgn, steps in ((0, 1, 2*r), (1, 1, 2*r), (0, -1, 2*r), (1, -1, 2*r)):
        dv = G.DIRS[mu]
        for _ in range(steps):
            if sgn > 0: W = W @ U[mu][n[0], n[1]]; n = [(n[0] + dv[0]) % L, (n[1] + dv[1]) % L]
            else: n = [(n[0] - dv[0]) % L, (n[1] - dv[1]) % L]; W = W @ U[mu][n[0], n[1]].T
    B = M.planes(P[base]); return float(np.linalg.det(B.T @ W @ B))

def run(L=64, d=16.0, ka=1.0, kb=2.0, Tr=40.0, T=60.0, dt=0.05, gam=0.0):
    P, U, X, Y = init(L, d); a = np.full((L, L), -1.0)
    g = G.Gauge(a, np.ones((3, L, L)), np.ones((2, L, L)), ka, kb)
    Pi = np.zeros_like(P); E = np.zeros_like(U); t = 0.0; rec = []
    while t < Tr + T - 1e-9:
        gg = 1.0 if t < Tr else gam
        P, Pi, U, E = g.step(P, Pi, U, E, dt, gam=gg); t += dt
        if abs(t*2 - round(t*2)) < dt/2 and (t >= Tr or round(t*2) % 4 == 0):
            e = g.edens(P, Pi, U, E); e0 = g.edens(np.broadcast_to(np.diag([M.U_]*3 + [M.V_]*2), P.shape), 0*Pi, np.broadcast_to(np.eye(5), U.shape), 0*E)
            ex = e - e0; cl = cores(g, U, X, Y, L)
            j0 = L//2; i1 = int(round(X.mean() - d/2 - 0.5*j0)); i2 = int(round(X.mean() + d/2 - 0.5*j0)); r = max(2, int(d//2) - 2)
            wl = [round(wilson(P, U, i, j0, r, L), 3) for i in (i1, i2)] + [round(wilson(P, U, (i1 + i2)//2, j0, int(d//2) + 6, L), 3)]
            w_ = np.clip(ex, 0, None); w_ = np.where(w_ > 0.2*w_.max(), w_, 0); xm = X.mean()
            pos = [float((w_*X)[side].sum()/max(w_[side].sum(), 1e-30)) for side in (X < xm, X >= xm)]
            rec.append(dict(t=round(t, 1), E=float(ex.sum()), emax=float(ex.max()), W=wl, pos=pos, n=len(cl), cl=[(round(c[0], 1), round(c[1], 1), round(c[2], 2)) for c in cl[:4]]))
            print(rec[-1], flush=True)
    name = f"gvort99_L{L}_d{d:g}_ka{ka:g}_kb{kb:g}_Tr{Tr:g}_g{gam:g}"
    json.dump(rec, open(name + ".json", "w")); np.savez(name + ".npz", P=P, U=U)
    return name, rec

if __name__ == "__main__":
    kw = {k: (int(v) if k == "L" else float(v)) for k, v in (a.split("=") for a in sys.argv[1:])}
    run(**kw)
