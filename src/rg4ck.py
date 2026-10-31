"""rg4ck (170): rungen4 with checkpoints (ck — interval in time units, ckdir — directory; resume is automatic).\nrungen4 (No. 60, stage B): the cell of the 3D Matrix is a tetrahedron (the simplex of its own dimension), not a triangle.
The mesh is tetrahedral (kind='tet', Kuhn subdivision: each cube = 6 tetrahedra x, x+e_a, x+e_a+e_b, x+(1,1,1)).
Cell energy — loops through all 4 vertices of the tetrahedron (3 quadrilateral loops per tetrahedron; the 2D analogue is the loop through the 3 vertices of a triangle).
For the vacuum to have the same amplitude w₊ as in 2D, the loop term has degree 3 in the amplitude: −β′ Re(P)|P|^(a−1), a = 3/k (k = 4 links);
ε₀ violation: Et·(|P|^a − Re(P)|P|^(a−1)). Normalisation "as in 2D" (variant C): link β = 2β/(a·k_e) (k_e — loops per link),
ε₀ × 8/(loops per node) (ε₀ weight per node as in 2D), κ and λ_c × 6/(links per node). The rest (α, local law, τ, diagnostics) — as in rungen.
The mesh triangles are kept only for diagnostics (|hol.|, flat nodes) and for the neighbourhood of the local law."""
import os; os.environ["NB"] = "0"; os.environ.setdefault("DELTA", "1e-6")
import numpy as np, json, sys, itertools
import fabgen as G, simNlib as N, local60 as LC, rnd51 as R, tile48 as TL, birth48 as BR
g = mu = 1.0

def quads(f):
    """Quadrilateral loops of the Kuhn tetrahedra: vertices (a, b, c, d) in traversal order; 3 per tetrahedron, 6 tetrahedra per node."""
    L = f.L; I = f.ijk; idx = lambda c: ((c[..., 0] % L)*L + (c[..., 1] % L))*L + (c[..., 2] % L); Q = []
    for p in itertools.permutations(range(3)):
        e0 = np.zeros(3, int); e0[p[0]] = 1; e1 = e0.copy(); e1[p[1]] = 1
        a, b, c, d = np.arange(f.n), idx(I + e0), idx(I + e1), idx(I + 1)
        Q += [np.stack(v, 1) for v in ((a, b, c, d), (a, b, d, c), (a, c, b, d))]
    return np.concatenate(Q)

class MQ:
    """As sim34d.M34, but the cells are loops of length k (here 4) with power a = 3/k."""
    def __init__(s, f, QV, beta, eps0, h, kappa, wp):
        s.f = f; s.QV = QV; s.k = QV.shape[1]; s.a = 3.0/s.k; s.eps0, s.h, s.kappa, s.wp = eps0, h, kappa, wp; s.d2 = float(os.environ["DELTA"])**2
        s.beta = beta; s.pin = np.zeros(f.m, bool)
    def links(s):
        """Link indices of the loops and signs (+1 if the link is stored in the traversal direction) — after flipping the arrows."""
        f = s.f; n = f.n; key = np.r_[f.tail*n + f.head, f.head*n + f.tail]; eid = np.r_[np.arange(f.m), np.arange(f.m)]
        sg = np.r_[np.ones(f.m, int), -np.ones(f.m, int)]; o = np.argsort(key); key, eid, sg = key[o], eid[o], sg[o]
        s.E = []; s.S = []
        for j in range(s.k):
            q = s.QV[:, j]*n + s.QV[:, (j + 1) % s.k]; pos = np.searchsorted(key, q); assert (key[pos] == q).all()
            s.E.append(eid[pos]); s.S.append(sg[pos])
        s.ke = np.bincount(np.concatenate(s.E), None, f.m)
    def unpack(s, y):
        m = s.f.m; return y[:m] + 1j*y[m:2*m], y[2*m:]
    def pack(s, z, psi): return np.concatenate([z.real, z.imag, psi])
    def loop(s, z):
        l = [np.where(S > 0, z[E], np.conj(z[E])) for E, S in zip(s.E, s.S)]; P = np.prod(l, 0); return l, P
    def rhs(s, t, y):
        f = s.f; z, psi = s.unpack(y); a_, b_ = f.tail, f.head; a = s.a
        A = np.abs(z); Ad = np.sqrt(A**2 + s.d2); chi = ((1 + psi)/2)**2; dchi = (1 + psi)/2
        we = Ad/s.wp; dpsi = psi[b_] - psi[a_]
        force = s.kappa*(np.bincount(a_, we*dpsi, f.n) - np.bincount(b_, we*dpsi, f.n)) - (psi**3 - psi) + s.h
        l, P = s.loop(z); aP = np.sqrt(np.abs(P)**2 + s.d2); pa = aP**(a - 1); rr = P.real*P/aP**2     # rr = Re P / P̄
        Et = s.eps0*np.mean([chi[s.QV[:, j]] for j in range(s.k)], 0)
        cB = pa*(1 + (a - 1)*rr); cE = -Et*pa*(a*P/aP - 1 - (a - 1)*rr)           # force on a link of the loop = Q̄·(β′ cB + cE)
        lw = np.zeros(f.m, complex); br = np.zeros(f.m, complex)
        for j in range(s.k):
            Qb = np.conj(np.prod([l[i] for i in range(s.k) if i != j], 0))
            for out, c in ((lw, Qb*cB), (br, Qb*cE)):
                v = np.where(s.S[j] > 0, c, np.conj(c)); out += np.bincount(s.E[j], v.real, f.m) + 1j*np.bincount(s.E[j], v.imag, f.m)
        cost = (aP**a - P.real*pa)/s.k*s.eps0
        for j in range(s.k): nd = s.QV[:, j]; force -= np.bincount(nd, dchi[nd]*cost, f.n)
        dz = -g*z - mu*A**2*z + s.beta*lw + br - 0.5*s.kappa*dpsi**2/s.wp*z/Ad
        dz[s.pin] = 0.0
        return np.concatenate([dz.real, dz.imag, force])
    def energy(s, z, psi):
        f = s.f; A = np.abs(z); chi = ((1 + psi)/2)**2; l, P = s.loop(z); aP = np.sqrt(np.abs(P)**2 + s.d2); pa = aP**(s.a - 1)
        Et = s.eps0*np.mean([chi[s.QV[:, j]] for j in range(s.k)], 0); bl = np.zeros(f.m)
        # β′ per link: a loop shares its contribution equally between its links (energetically consistent only for constant β′)
        return float(np.sum(g*A**2/2 + mu*A**4/4) - np.mean(s.beta)*np.sum(P.real*pa) + np.sum(Et*(aP**s.a - P.real*pa))
                     + np.sum((psi**2 - 1)**2/4 - s.h*psi) + s.kappa/2*np.sum(np.sqrt(A**2 + s.d2)/s.wp*(psi[f.tail] - psi[f.head])**2))

def run(ck=5.0, ckdir="ck170", L=16, seed=1, Tt=100.0, dt=0.005, n=100, ell=3.0, g_=0.02, s0=3.5, h=0.3, lc=1.3, beta=1.5, eps0=3.0, kappa=1.0, ac=1.0, check=0, init="", pr=0.0, fp=0.0):
    f = G.FabricG(L, "tet"); dg = float(f.deg[0]); wp = N.wplus(beta)
    s = LC.Loc.__new__(LC.Loc); s.fab = f
    tau = G.tau_cubic(L, ell, seed); Gg = g_*(f.disp() @ np.array([1.0, 0.31, 0.17])); bad = tau[f.head] - tau[f.tail] + Gg < 0; R.flip(f, bad)
    QV = quads(f); tq = np.bincount(QV.ravel(), None, f.n)[0]
    mod = MQ(f, QV, 0.0, eps0*8/tq, h, kappa*6/dg, wp); mod.links(); mod.beta = 2*beta/(mod.a*mod.ke)
    s.mod = mod; s.wp = wp; s.lc = lc*6/dg; s.cs, s.cc = 1.0 + 0j, 1 + 1j*ac
    P = G.local_P(f); s.P = P; s.n = n; s.tc = 1.0; s.K = G.fft_K(f, P, n)
    nb = [[] for _ in range(f.n)]
    for p, q in zip(f.tail, f.head): nb[p].append(q); nb[q].append(p)
    rng = np.random.default_rng(seed); r_ = np.roots([1, 0, -1, -h]); psiM = r_[np.isreal(r_)].real.min()
    z = np.sqrt(s0)*(1 + 0.01*(rng.standard_normal(f.m) + 1j*rng.standard_normal(f.m))); psi = psiM + 0.01*rng.standard_normal(f.n)
    if check: return s, f, mod, z, psi
    if init:                                                             # planting (as rungen (114)): a ψ ball of radius pr at the centre of the largest flat patch of the init snapshot
        d0 = np.load(init + ".npz"); assert (d0["bad"] == bad).all(); z = d0["z"]; psi = d0["psi"].copy()
        hol = np.abs(f.holonomy(z)); hn = np.bincount(f.TV.ravel(), np.repeat(hol, 3), f.n)/f.tnode; M = (hn < 0.35) & (psi < 0)
        cs_ = []; seen = np.zeros(f.n, bool)
        for v in np.flatnonzero(M):
            if seen[v]: continue
            comp = [v]; seen[v] = True; st = [v]
            while st:
                u = st.pop()
                for w_ in nb[u]:
                    if M[w_] and not seen[w_]: seen[w_] = True; comp.append(w_); st.append(w_)
            cs_.append(comp)
        big_ = max(cs_, key=len); P3 = f.ijk[big_]
        dd = (P3[:, None, :] - P3[None, :, :] + L//2) % L - L//2; c0 = big_[int(np.argmax((np.abs(dd).max(2) <= 3).sum(1)))]
        dI = (f.ijk - f.ijk[c0] + L//2) % L - L//2; rr = np.linalg.norm(dI, axis=1); psi[rr <= pr] = r_[np.isreal(r_)].real.max(); zone = rr <= pr
        print("plant at", f.ijk[c0], "patch", len(big_), "nodes", int((rr <= pr).sum()), flush=True)
    s.S0 = float((np.abs(z)**2).sum()); s.m0 = s.S0/f.m
    name = f"rq_L{L}_s{seed}_n{n}" + (f"_ac{ac:g}" if ac != 1.0 else "") + (f"_e{eps0:g}" if eps0 != 3.0 else "") + (f"_lc{lc:g}" if lc != 1.3 else "") + (f"_pr{pr:g}" if init else "")
    t = 0.0; i = 0; ST = int(round(1/dt)); w2 = []; prev = None; FP = int(round(fp/dt)) if fp else 0; bal = []
    os.makedirs(ckdir, exist_ok=True); CK = os.path.join(ckdir, name + "_ck"); CKS = int(round(ck/dt)) if ck else 0
    if CKS and os.path.exists(CK + ".npz"):                              # resume from a checkpoint
        d = np.load(CK + ".npz"); z = d["z"]; psi = d["psi"]; t = float(d["t"]); i = int(d["i"]); prev = d["prev"] if d["prev"].size else None
        w2 = json.load(open(CK + ".json"))["w2"]; print("resume from t =", t, flush=True)
    if FP: name = "bal_" + name
    print("loops/node", tq, "loops/link", np.unique(mod.ke), flush=True)
    while t < Tt - 1e-9:
        if FP and i % FP == 0:                                           # components of the force on ψ in the planted ball (as rungen fp, (115))
            Pp = s.F(z, psi, parts=True); _, Pq = mod.loop(z); qin = zone[QV].all(1); inn = zone[f.tail] & zone[f.head]
            r = dict(t=round(t, 3), psi=float(psi[zone].mean()), A2=float((np.abs(z[inn])**2).mean()), qhol=float(np.abs(np.angle(Pq[qin])).mean()))
            r.update({k: float(v[zone].mean()) for k, v in Pp.items()}); bal.append(r); print(r, flush=True)
        z, psi = s.step(z, psi, dt); t += dt; i += 1
        if not np.isfinite(psi).all() or not np.isfinite(z).all(): w2.append(dict(t=round(t, 2), blowup=True)); break
        if i % ST == 0:
            rec, prev = TL.stats(f, z, prev); B = psi > 0; cs = BR.comps(f, B, nb) if B.any() else []
            hol = np.abs(f.holonomy(z)); hn = np.bincount(f.TV.ravel(), np.repeat(hol, 3), f.n)/f.tnode
            A2n = (np.bincount(f.tail, np.abs(z)**2, f.n) + np.bincount(f.head, np.abs(z)**2, f.n))/f.deg
            _, Pq = mod.loop(z); qh = np.abs(np.angle(Pq)); qn = np.bincount(QV.ravel(), np.repeat(qh, 4), f.n)/tq
            rec.update(t=round(t, 1), nB=int(B.sum()), nuc=sum(1 for q in cs if q >= 5), big=cs[0] if cs else 0, top=cs[:6],
                       flat=float((hn < 0.35).mean()), proto=float(((hn < 0.35) & (A2n > 5)).mean()), holm=float(hn.mean()),
                       qhol=float(qh.mean()), qflat=float((qn < 0.35).mean()))
            w2.append(rec); print(rec, flush=True)
        if CKS and i % CKS == 0:
            np.savez(CK + "_tmp.npz", z=z, psi=psi, t=t, i=i, prev=prev if prev is not None else np.zeros(0)); os.replace(CK + "_tmp.npz", CK + ".npz")
            json.dump(dict(w2=w2), open(CK + "_tmp.json", "w")); os.replace(CK + "_tmp.json", CK + ".json")
    np.savez(name + ".npz", z=z, psi=psi, bad=bad)
    json.dump(dict(name=name, kind="tet-quad", L=L, nloop=len(QV), m=f.m, w2=w2, **({"bal": bal} if FP else {})), open(name + ".json", "w")); return name
if __name__ == "__main__":
    kw = {k: (v if k in ("init", "ckdir") else (int(v) if k in ("L", "seed", "n", "check") else float(v))) for k, v in (a.split("=") for a in sys.argv[1:])}
    print(run(**kw))
