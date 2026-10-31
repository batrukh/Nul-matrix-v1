"""local60: local law of intensity conservation (sec. 4.4 (60), queue No. 50). As cons58 (kind=almost, L48, free ψ), but without a global μ.
The model itself creates/destroys link intensity at the rate s_e = 2 Re(z̄ ż)_e. Local law: this gain/loss is compensated from the neighbourhood of the link,
dρ_e/dt = s_e − (K s)_e, where K = P^n, P = (I + A)/5 (A — adjacency of links through a common triangle; on the regular lattice P is doubly stochastic,
so Σρ is conserved exactly). Neighbourhood radius ≈ √n links; n → ∞ — the global law (58). Implementation: ż_e −= (K s)_e/(2ρ_e)·z_e (ρ_e ≥ 0.05)."""
import numpy as np, sys, json, os
import scipy.sparse as sp
os.environ.setdefault("DELTA", "1e-6")
NBON = os.environ.get("NB", "1") == "1"                                   # compiled force (nbforce), the same formula
import nbforce as NB, fftk as FK, stall43 as T, rnd51 as R, tile48 as TL, birth48 as BR, seam51 as SM

class Loc(T.Sys):
    def F(s, z, psi, parts=False):
        if parts: return super().F(z, psi, parts)
        dz, dpsi = NB.force(s, z, psi) if NBON else super().F(z, psi); rho = np.maximum(np.abs(z)**2, 0.05)
        src = 2*np.real(np.conj(z)*dz)
        if getattr(s, 'K', None) is not None: ks = s.K(src)                 # P^n via FFT (fftk), exactly the same
        else:
            ks = src
            for _ in range(s.n): ks = s.P @ ks
        ks = ks + (rho.sum() - s.S0)/(s.tc*len(rho))                      # weak correction of the numerical drift
        return dz - ks/(2*rho)*z, dpsi

class LocR(Loc):                                                          # No. 60: the same model on a random triangulation (rnd51)
    __init__ = R.Sys.__init__

def tau_field(L, ell, seed, iso=0):
    """Smooth random periodic τ with scale ℓ (as tau52.tau_field, mode rand): Gaussian filter of white noise."""
    rng = np.random.default_rng(700 + seed); w = rng.standard_normal((L, L))
    kk = np.fft.fftfreq(L)*2*np.pi; K2 = kk[:, None]**2 + kk[None, :]**2
    if iso: K2 = (4/3)*(kk[:, None]**2 - kk[:, None]*kk[None, :] + kk[None, :]**2)   # (79): isotropic in the Cartesian embedding of the triangular lattice
    t = np.real(np.fft.ifft2(np.fft.fft2(w)*np.exp(-K2*ell**2/2))); return t.ravel()/t.std()

def vortex_cores(L, k, seed):
    """k pairs of pre-time vortices: charges +1 and −1, centres inside triangles (shift 1/3), random places (rng 800 + seed)."""
    rng = np.random.default_rng(800 + seed); c = rng.integers(0, L, (2*k, 2)) + 1/3
    return [(float(a), float(b), 1 if i % 2 == 0 else -1) for i, (a, b) in enumerate(c)]

def vortex_dtheta(f, L, cores, M=3):
    """Increment of the cyclic pre-time θ along each link tail→head: Σ q·Δarg over the vortices and their images on the torus (|m|,|n| ≤ M)."""
    d = R.disp(f) if hasattr(R, "disp") else None
    pt = f.xy[f.tail].astype(float); x_, y_ = pt.T; x2, y2 = f.xy[f.head].T
    dx = (x2 - x_ + L//2) % L - L//2; dy = (y2 - y_ + L//2) % L - L//2; ph = pt + np.stack([dx, dy], 1)
    out = np.zeros(f.m)
    for (a, b, q) in cores:
        for m in range(-M, M + 1):
            for n in range(-M, M + 1):
                ca, cb = a + m*L, b + n*L
                d_ = np.arctan2(ph[:, 1] - cb, ph[:, 0] - ca) - np.arctan2(pt[:, 1] - cb, pt[:, 0] - ca)
                out += q*((d_ + np.pi) % (2*np.pi) - np.pi)
    return out

def run(L=48, ln=46, s0=4.0, n=10, Tt=400.0, seed=1, h=0.3, dt=0.005, lc=1.3, tc=1.0, cont=0.0, k=1, ell=0.0, D=0.0, g=0.0, rnd=0.0, gseed=0, ac=1.0, cut=0, h13=0.0, h48=0.0, ph=0, vort=0, ga=999.0, iso=0, sm=0.0, mi=0.0, Tk=0.0, pf=0.0, pg=0.2, beta=1.5, eps0=3.0, kappa=1.0, pe=0.0, pseed=0, gz=0, gv=0.0, gfl=0.02, sn=0, gk=5, gd=0, pb=0.0, sk=0, snp=0, nz=0.01, pnz=0.01, jin=0.0, kb=0.0, kl=0, ck=0, ckdir='', pR=0.0, pq=0.0, rs=''):
    rng = np.random.default_rng(seed); r_ = np.roots([1, 0, -1, -h]); psiM = r_[np.isreal(r_)].real.min()
    s = LocR(L=L, gseed=gseed, jit=rnd, h=h, lc=lc) if rnd else Loc(L=L, h=h, lc=lc, ac=ac, beta=beta, eps0=eps0, kappa=kappa); f = s.fab
    if ell:                                             # No. 42: arrows from a smooth random τ (as tau52), without hand-made seams
        tau = tau_field(L, ell, seed, iso)
        if rnd:                                          # No. 60: τ at a node — bilinear interpolation of the periodic τ (no equal τ at neighbours)
            T2 = tau.reshape(L, L); px, py = f.pos[:, 0] % L, f.pos[:, 1] % L
            if g: px = (px - py/2) % L                   # (83): with tilt — proper index coordinates (pos = (i + j/2, j)); without tilt as in (70)–(81)
            i0, j0 = np.floor(px).astype(int), np.floor(py).astype(int)
            u, v = px - i0, py - j0; i1, j1 = (i0 + 1) % L, (j0 + 1) % L
            tau = (1-u)*(1-v)*T2[i0, j0] + u*(1-v)*T2[i1, j0] + (1-u)*v*T2[i0, j1] + u*v*T2[i1, j1]
        #                                                No. 58: τ = φ + g·(x + 0.31·y) — tilt without a discontinuity (differences along links, no torus wrapping)
        x_, y_ = f.xy[f.tail].T; x2, y2 = f.xy[f.head].T
        Gg = g*(((x2 - x_ + L//2) % L - L//2) + 0.31*((y2 - y_ + L//2) % L - L//2))    # contribution of the tilt to Δτ of the link tail→head
        if rnd and g:                                    # (83): on the random mesh the tilt is taken from the true node offsets (di = dX − dY/2, dj = dY)
            dd_ = R.disp(f); Gg = g*((dd_[:, 0] - dd_[:, 1]/2) + 0.31*dd_[:, 1])
        if ga != 999.0:                                  # No. 62: tilt at an angle ga° in the Cartesian embedding (e1 = (1,0), e2 = (1/2, √3/2)); |w| as in the default (1.024)
            ddx = (x2 - x_ + L//2) % L - L//2; ddy = (y2 - y_ + L//2) % L - L//2; th = np.radians(ga)
            Gg = g*1.024*((ddx + 0.5*ddy)*np.cos(th) + (np.sqrt(3)/2*ddy)*np.sin(th))
        if sm:                                           # (82): smooth arrow — links are not flipped; rotation 1 + iα_e, α_e = α·tanh(Δτ_e/sm)
            dT = tau[f.head] - tau[f.tail] + Gg; bad = dT < 0; s.acv = ac*np.tanh(dT/sm)   # sm → 0 — the same as the binary arrow (z → z̄)
        else:
            bad = tau[f.head] - tau[f.tail] + Gg < 0; R.flip(f, bad); Gg = np.where(bad, -Gg, Gg)
    elif vort:                                          # (77): cyclic pre-time — vort pairs of vortices (+1/−1) + tilt; arrow loops only in the vortex cores
        tau = None; cores = vortex_cores(L, vort, seed); dth = vortex_dtheta(f, L, cores)
        x_, y_ = f.xy[f.tail].T; x2, y2 = f.xy[f.head].T
        Gg = g*(((x2 - x_ + L//2) % L - L//2) + 0.31*((y2 - y_ + L//2) % L - L//2))
        bad = dth + Gg < 0; R.flip(f, bad); s.cores = cores
    else:
        R.flip(f, SM.seams(f))
        xa, ya = f.xy[f.tail].T; xb, yb = f.xy[f.head].T
        dy = (yb - ya + L//2) % L - L//2; dx = (xb - xa + L//2) % L - L//2; my = np.minimum(ya, yb)
        H = (dy == 0) & (np.abs(dx) == 1); V = (dx == 0) & (np.abs(dy) == 1); c = L//2
        cs_ = [(2*i + 1)*L//(2*k) for i in range(k)]      # k rows and k columns of seams: k*k crosses (k=1 — one cross, as before)
        bad = np.zeros(f.m, bool)
        for c in cs_: bad |= H & (ya == c) | V & (xa == c) & (my >= c - ln//2) & (my < c - ln//2 + ln)
        c = L//2; mx = np.minimum(xa, xb)
        if sk:                                           # No. 53: open and closed seams (sk=1 — one row segment of length ln; 2 — cross of two segments ln; 3 — closed loop ln×ln; 4 — no seams; 5 — cross H + third family, No. 46)
            Hs = H & (ya == c) & (mx >= c - ln//2) & (mx < c - ln//2 + ln); Vs = V & (xa == c) & (my >= c - ln//2) & (my < c - ln//2 + ln)
            x0 = c - ln//2
            Lp = H & np.isin(ya, (x0, x0 + ln)) & (mx >= x0) & (mx < x0 + ln) | V & np.isin(xa, (x0, x0 + ln)) & (my >= x0) & (my < x0 + ln)
            Dd = (dx == -dy) & (np.abs(dx) == 1); Ds = Dd & ((xa + ya) % L == (2*c) % L) & (my >= c - ln//2) & (my < c - ln//2 + ln)   # No. 46: seam along the third family (x + y = 2c)
            bad = {1: Hs, 2: Hs | Vs, 3: Lp, 4: np.zeros(f.m, bool), 5: Hs | Ds}[sk]   # 5 — cross H + third family
        R.flip(f, bad)
    E3 = np.stack([f.E1, f.E2, f.E3], 1); rows = []; cols = []
    for a, b in ((0, 1), (1, 2), (0, 2)): rows += [E3[:, a], E3[:, b]]; cols += [E3[:, b], E3[:, a]]
    A = sp.csr_matrix((np.ones(sum(len(r) for r in rows)), (np.concatenate(rows), np.concatenate(cols))), shape=(f.m, f.m))
    A.data[:] = 1.0; s.P = ((sp.identity(f.m) + A)/5.0).tocsr(); s.n = n; s.tc = tc
    assert np.allclose(np.asarray(s.P.sum(0)).ravel(), 1.0)
    s.K = None if rnd else FK.build(f, L, s.P, n)          # FFT only on the regular lattice; on the random mesh — multiplication by P
    nb = [[] for _ in range(f.n)]
    for p, q in zip(f.tail, f.head): nb[p].append(q); nb[q].append(p)
    degn = np.array([len(v) for v in nb])
    z = np.sqrt(s0)*(1 + nz*(rng.standard_normal(f.m) + 1j*rng.standard_normal(f.m))); psi = psiM + pnz*rng.standard_normal(f.n)   # No. 53: nz, pnz — amplitude of the initial noise of links
    if pR:                                               # (177): local hole in the canvas — inside a circle of radius pR the link intensity is lowered by the fraction pq, the removed part is given evenly to the rest (same budget)
        X_ = f.xy[:, 0] + 0.5*f.xy[:, 1]; Y_ = np.sqrt(3)/2*f.xy[:, 1]; c_ = (L//2 + 0.5*(L//2), np.sqrt(3)/2*(L//2))
        dmin = None
        for a_ in (-1, 0, 1):
            for b_ in (-1, 0, 1):
                d_ = np.hypot(X_ - c_[0] + a_*L + b_*L/2, Y_ - c_[1] + b_*np.sqrt(3)/2*L); dmin = d_ if dmin is None else np.minimum(dmin, d_)
        hole = (dmin[f.tail] <= pR) & (dmin[f.head] <= pR); Az = np.abs(z)**2; rem = float(Az[hole].sum())*pq
        z = np.where(hole, z*np.sqrt(1 - pq), z*np.sqrt(1 + rem/float(Az[~hole].sum())))
        s.hole_nodes = np.where(dmin <= pR)[0]; s.hole_d = dmin; print("hole", int(hole.sum()), "threads, removed", round(rem, 1), flush=True)
    if sm: z = np.where(bad, np.conj(z), z)              # (82): the same realisation as with the binary arrow (z is written in the reference direction of the link)
    s.S0 = float((np.abs(z)**2).sum()); s.m0 = s.S0/f.m
    t = 0.0; i = 0; w2 = []; prev = None; spl = []
    name = f"lc_s0{s0:g}_n{n}_L{L}_s{seed}" + (f"_k{k}" if k > 1 else "") + (f"_tau{ell:g}" if ell else "") + (f"_D{D:g}" if D else "") + (f"_g{g:g}" if g else "") + (f"_rnd{rnd:g}g{gseed}" if rnd else "") + (f"_ac{ac:g}" if ac != 1.0 else "") + (f"_v{vort}g{g:g}" if vort else "") + (f"_ga{ga:g}" if ga != 999.0 else "") + ("_iso" if iso else "") + (f"_sm{sm:g}" if sm else "") + (f"_sk{sk}l{ln}" if sk else "") + (f"_nz{nz:g}" if nz != 0.01 else "") + (f"_pnz{pnz:g}" if pnz != 0.01 else "") + (f"_fb{jin:g}k{kb:g}" if kb else "") + ("_kl" if kl else "") + (f"_pR{pR:g}q{pq:g}" if pR else "") + (("_rs" + os.path.basename(rs)[:-4]) if rs else "")
    assert not (cont and sm)
    ST = int(round(1.0/dt))                                               # (85): output and τ step — once per unit of time for any dt (dt = 0.005 → 200, as before)
    name += (f"_b{beta:g}" if beta != 1.5 else "") + (f"_e{eps0:g}" if eps0 != 3.0 else "") + (f"_kp{kappa:g}" if kappa != 1.0 else "") + (f"_h{h:g}" if h != 0.3 else "") + (f"_lc{lc:g}" if lc != 1.3 else "")   # No. 67: sensitivity to the constants
    name += (f"_dt{dt:g}" if dt != 0.005 else "") + (f"_mi{mi:g}r" if mi else "") + (f"_pf{pf:g}" + (f"g{pg:g}" if pg != 0.2 else "") + (f"e{pe:g}" if pe else "") + (f"p{pseed}" if pseed else "") + (f"_gz{gfl:g}" if gz else "") + (f"_gv{gv:g}" if gv else "") if pf else "")
    if pf:                                               # (92): live protofield Φ★ (phi90) at the nodes, coupled to ψ: energy pf·a(ψ)/2·trΦ², a = 1 − 2χ(ψ)
        import phi90 as PH                               # Φ̈ + γΦ̇ = ΔΦ − ∂V/∂Φ (V with a(ψ)), ψ̇ += pf·χ'(ψ)·trΦ² (back-reaction of the protofield on the domain)
        Phi = PH.sym0(0.01*np.random.default_rng(700 + seed + 1000*pseed).standard_normal((L, L, 5, 5))); Vphi = np.zeros_like(Phi)
        Bh = PH.sym0(np.random.default_rng(900).standard_normal((5, 5))); Bh /= np.linalg.norm(Bh)   # (103): heredity — a constant direction from the Matrix, strength pe
        snaps = []                                       # (103): snapshots of ψ and Φ★ every 5 units of time
        chi_ = lambda p_: 0.5*(1 + np.tanh(p_/0.2)); dchi_ = lambda p_: (1 - np.tanh(p_/0.2)**2)/0.4
        def phistep(Phi, Vphi, psi):
            aa = (1.0 - 2.0*chi_(psi)).reshape(L, L)
            A_ = PH.lap(Phi) - PH.dV(Phi, aa, pb) + (pe*Bh if pe else 0)
            Vphi = (Vphi + dt*(A_ - pg*Vphi)); Phi = PH.sym0(Phi + dt*Vphi)            # semi-implicit Euler (symplectic at γ = 0)
            t2 = np.trace(Phi @ Phi, axis1=-2, axis2=-1).reshape(-1)
            return Phi, Vphi, psi + dt*pf*dchi_(psi)*t2
    if pf and gz:                                        # (104): gauge field together with the protofield in the live model (gf104), one step per 10 link steps
        import gf104 as GFm, matter98 as M98, scipy.ndimage as ndi
        GF = GFm.GF(L, seed + 1000*pseed, floor=gfl, pg=pg, b=pb); KG = 10; gtrack = []; gplant = None
        Xg, Yg = PH.coords(L)
    assert not (pf and (mi or rnd or cont))
    assert not (mi and (sm or cont))
    def stepph(z, w, psi):                                # (88): inertia of the phases only: Ȧ/A = Re(F/z), m·θ̈ + θ̇ = Im(F/z)
        def fr(z_, w_, p_):
            Fz, Fp = s.F(z_, p_); u = Fz*np.conj(z_)/(np.abs(z_)**2 + 0.01); return z_*(u.real + 1j*w_) + (Fz - z_*u)*(np.abs(z_)**2 < 0.25), (u.imag - w_)/mi, Fp   # the phase near |z| = 0 is undefined: regularisation 0.01; for |z| < 0.5 the link moves by the old law
        k1 = fr(z, w, psi); k2 = fr(z + dt/2*k1[0], w + dt/2*k1[1], psi + dt/2*k1[2])
        k3 = fr(z + dt/2*k2[0], w + dt/2*k2[1], psi + dt/2*k2[2]); k4 = fr(z + dt*k3[0], w + dt*k3[1], psi + dt*k3[2])
        return (z + dt/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0]), w + dt/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1]), psi + dt/6*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2]))
    wv = (s.F(z, psi)[0]/z).imag if mi else None
    if cont:                                                              # continuation from t = cont: the same state, the same S0, P, defects
        d0 = np.load(name + ".npz"); assert (d0["bad"] == bad).all()
        z, psi = d0["z"], d0["psi"]; t = float(cont); i = int(round(cont/dt)); name += f"_T{Tt:g}"
        if D:                                          # No. 61: τ is reproduced up to t = cont (the spreading is deterministic and does not depend on z)
            for _ in range(i//ST):
                ns = int(np.ceil(12*D))
                for _ in range(ns):
                    g_ = tau[f.head] - tau[f.tail]
                    tau = tau + (D*ST*dt/ns)*(np.bincount(f.tail, g_, f.n) - np.bincount(f.head, g_, f.n))
                fl = tau[f.head] - tau[f.tail] + Gg < 0; R.flip(f, fl); Gg = np.where(fl, -Gg, Gg)
        if cut:                                        # No. 61: the largest domain is cut down to cut nodes (graph "ball" around the deepest node)
            B = psi > 0; lab = -np.ones(f.n, int); k_ = 0
            for v in np.where(B)[0]:
                if lab[v] >= 0: continue
                st = [v]; lab[v] = k_
                while st:
                    u = st.pop()
                    for w_ in nb[u]:
                        if B[w_] and lab[w_] < 0: lab[w_] = k_; st.append(w_)
                k_ += 1
            big_ = np.argmax(np.bincount(lab[lab >= 0])); W = lab == big_
            def bfs(src, allow):
                dd = -np.ones(f.n, int); dd[src] = 0; q = list(src)
                while q:
                    q2 = []
                    for u in q:
                        for w_ in nb[u]:
                            if allow[w_] and dd[w_] < 0: dd[w_] = dd[u] + 1; q2.append(w_)
                    q = q2
                return dd
            edge_ = [v for v in np.where(W)[0] if any(not W[w_] for w_ in nb[v])]
            dep = bfs(edge_, W); c0 = int(np.argmax(dep)); dc = bfs([c0], W)
            keep = np.zeros(f.n, bool); order = np.argsort(np.where(dc >= 0, dc, 10**9) + 1e-6*np.arange(f.n)); keep[order[:cut]] = True
            rm = W & ~keep; Aa = np.abs(z)**2; o = (psi[f.tail] < 0) & (psi[f.head] < 0); Ao = Aa[o].mean()
            psi = np.where(rm, psiM, psi); ze = rm[f.tail] | rm[f.head]
            z = np.where(ze, z*np.sqrt(Ao/np.maximum(Aa, 1e-12)), z)        # links of the removed part — to the mean intensity outside domains
            if ph:                                     # (75): "clean" cut — the phases of links in the removed part are random (the domain's memory is erased)
                z = np.where(ze, np.abs(z)*np.exp(2j*np.pi*np.random.default_rng(900 + seed).random(f.m)), z); name += "_ph"
            print("cut", int(W.sum()), "->", int(keep.sum()), "depth", int(dep.max()), "Aout", round(float(Ao), 3), flush=True)
            name += f"_cut{cut}"
            if h13 or h48:                             # (75): transfer the intensity profile around the domain (rings 1–3 and 4–8 from the domain), ring 9–15 compensates the sum
                dw = bfs(list(np.where(keep)[0]), np.ones(f.n, bool)); de = np.maximum(dw[f.tail], dw[f.head]); Aa = np.abs(z)**2
                sc = np.ones(f.m); tot0 = Aa.sum()
                for (a_, b_), v_ in (((1, 3), h13), ((4, 8), h48)):
                    m_ = (de >= a_) & (de <= b_)
                    if v_: sc[m_] = v_/Aa[m_].mean()
                m9 = (de >= 9) & (de <= 15); A2 = Aa*sc; sc[m9] *= 1 + (tot0 - A2.sum())/A2[m9].sum()
                z = z*np.sqrt(sc); print("halo", h13, h48, "sum", round(float((np.abs(z)**2).sum()/tot0), 6), flush=True)
                name += f"_h{h13:g}-{h48:g}"
    snl = []; snv = []
    if sn: name += "_sn"
    if gk != 5: name += f"_gk{gk}"
    if gd: name += "_gd"
    if pb: name += f"_pb{pb:g}"                        # (123): cubic term b of the Φ★ potential
    ckf = os.path.join(ckdir or ".", os.path.basename(name) + "_ck.pkl") if ck else None   # (166): checkpoint every ck units of time, resume from it
    if ckf and os.path.exists(ckf):
        import pickle; C = pickle.load(open(ckf, "rb"))
        z, psi, t, i, bad, prev, w2, s.S0, s.m0 = C["z"], C["psi"], C["t"], C["i"], C["bad"], C["prev"], C["w2"], C["S0"], C["m0"]; snl, snv = C.get("snl", snl), C.get("snv", snv)
        f.tail, f.head, f.S1, f.S2, f.S3 = C["fab"]
        if D: tau, Gg = C["tau"], C["Gg"]
        if pf and gz: GF, gtrack, gplant = C["GF"], C["gtrack"], C["gplant"]; Phi = GF.P
        elif pf: Phi, Vphi = C["Phi"], C["Vphi"]
        print("resume", t, flush=True)
    if rs and not (ckf and os.path.exists(ckf)):                # (183): start from a prepared state (z, ψ, t) — the intervention in the island is done beforehand in the file
        R_ = np.load(rs); z = R_["z"].astype(complex); psi = R_["psi"].astype(float); t = float(R_["t"]); i = int(round(t/dt)); print("restart", rs, t, flush=True)
    while t < Tt - 1e-9:
        if ckf and i % (ck*ST) == 0 and i > 0:
            import pickle; C = dict(z=z, psi=psi, t=t, i=i, bad=bad, prev=prev, w2=w2, S0=s.S0, m0=s.m0, fab=(f.tail, f.head, f.S1, f.S2, f.S3)); C.update(snl=snl, snv=snv)
            if D: C.update(tau=tau, Gg=Gg)
            if pf and gz: C.update(GF=GF, gtrack=gtrack, gplant=gplant)
            elif pf: C.update(Phi=Phi, Vphi=Vphi)
            pickle.dump(C, open(ckf + ".tmp", "wb")); os.replace(ckf + ".tmp", ckf)
        if mi: z, wv, psi = stepph(z, wv, psi)
        else: z, psi = s.step(z, psi, dt)
        if pf and gz:
            if (i + 1) % KG == 0: psi = GF.step(psi, KG*dt, pf); Phi = GF.P
        elif pf: Phi, Vphi, psi = phistep(Phi, Vphi, psi)
        t += dt; i += 1
        if i % ST == 0:
            if D:                                         # No. 42 step 2: τ spreads (∂τ/∂t = D·Δτ at the nodes), arrows follow τ
                ns = int(np.ceil(12*D))
                for _ in range(ns):
                    g_ = tau[f.head] - tau[f.tail]
                    tau = tau + (D*ST*dt/ns)*(np.bincount(f.tail, g_, f.n) - np.bincount(f.head, g_, f.n))
                if sm:                                    # (82): the smooth arrow follows τ without flipping links
                    dT = tau[f.head] - tau[f.tail] + Gg; s.acv = ac*np.tanh(dT/sm); fl = (dT < 0) != bad; bad = dT < 0; nfl = int(fl.sum())
                    od = np.bincount(np.where(bad, f.head, f.tail), None, f.n); next_ = int(((od == 0) | (od == 6)).sum())
                    ndef = int((dT*Gg < 0).sum()) if g else None
                else:
                    fl = tau[f.head] - tau[f.tail] + Gg < 0; R.flip(f, fl); z = np.where(fl, np.conj(z), z)   # the link state is the same, only the arrow changed
                    if mi: wv = np.where(fl, -wv, wv)                                                      # (88): θ → −θ, hence θ̇ too
                    Gg = np.where(fl, -Gg, Gg); nfl = int(fl.sum())
                    od = np.bincount(f.tail, None, f.n); next_ = int(((od == 0) | (od == degn)).sum())         # extrema of τ: all arrows of a node point inwards or outwards
                    ndef = int((Gg < 0).sum()) if g else None                                              # links running against the tilt (mismatch with respect to the pure tilt)
            B = psi > 0; cs = BR.comps(f, B, nb) if B.any() else []
            Aa = np.abs(z)**2; w = (psi[f.tail] > 0) & (psi[f.head] > 0); o = (psi[f.tail] < 0) & (psi[f.head] < 0)
            prevA = prev; rec, prev = TL.stats(f, z, prev)
            if kb and prevA is not None:                  # (131): inflow of strain jin per link per unit time and loss kb·boiling (boiling dissipates strain); m₀ follows the budget
                dA = prev - prevA; bf = float(np.abs(dA - dA.mean()).mean())   # boiling without the uniform shift from the inflow/loss itself
                if kl:                                    # (132): local loss — each link loses kb·its own boiling |ΔA_e − ⟨ΔA⟩|, the inflow stays uniform
                    An = prev - np.minimum(kb*np.abs(dA - dA.mean())*ST*dt, 0.5*prev); z = z*np.sqrt(An/np.maximum(prev, 1e-12)); lost = float((prev - An).sum()); prev = An
                    s.S0 = max(s.S0 - lost + f.m*jin*ST*dt, 0.05*f.m)
                else: s.S0 = max(s.S0 + f.m*(jin - kb*bf)*ST*dt, 0.05*f.m)
                s.m0 = s.S0/f.m; rec.update(s0=round(s.m0, 4), bf=round(bf, 4))
            rec.update(t=round(t, 1), S=round(float(Aa.sum())/s.S0, 5), Ain=round(float(Aa[w].mean()), 3) if w.any() else None,
                       Aout=round(float(Aa[o].mean()), 3) if o.any() else None, nB=int(B.sum()), nuc=sum(1 for q_ in cs if q_ >= 5), big=cs[0] if cs else 0)
            if D: rec.update(nfl=nfl, next=next_, ndef=ndef)
            if pf and gz:
                if gv and gplant is None and t >= gv - 1e-9:     # planting a vortex at the deepest point of the domain
                    dist = ndi.distance_transform_edt(np.tile(psi.reshape(L, L) > 0, (3, 3)))[L:2*L, L:2*L]; i0, j0 = np.unravel_index(np.argmax(dist), dist.shape)   # periodic (3×3 tiling)
                    gplant = dict(t=round(t, 1), ij=(int(i0), int(j0)), depth=float(dist[i0, j0]), xy=GF.plant(i0, j0)); print("plant", gplant, flush=True)
                if i % (gk*ST) == 0:                       # (120): detector output interval gk units (default 5)
                    det, wm = GF.defects(psi, aware=bool(gd)); cl = M98.clusters(np.where((det < 0) & wm, -1.0, 1.0), Xg, Yg, L)
                    if gd:                                 # (120): old detector for comparison and the fraction of the domain in the "three positive" vacuum
                        det0, _ = GF.defects(psi); ncl_old = len(M98.clusters(np.where((det0 < 0) & wm, -1.0, 1.0), Xg, Yg, L))
                        np3 = (np.linalg.eigvalsh(GF.P) > 0).sum(-1) >= 3; Wn = psi.reshape(L, L) > 0.3; f3 = float(np3[Wn].mean()) if Wn.any() else None
                    near = None
                    if gplant:
                        cx, cy = gplant["xy"]; dd = [float(M98.wrapd(np.array(c[0]), np.array(c[1]), cx, cy, L)[0]) for c in cl]
                        near = round(min(dd), 1) if dd else None
                    dist_ = ndi.distance_transform_edt(np.tile(psi.reshape(L, L) > 0, (3, 3)))[L:2*L, L:2*L]   # (104b): depth of each vortex and domain
                    dep = []
                    for c in cl:
                        k_ = int(np.argmin((Xg - c[0])**2 + (Yg - c[1])**2)); dep.append(round(float(dist_.reshape(-1)[k_]), 1))
                    gtrack.append(dict(t=round(t, 1), ncl=len(cl), cl=cl, wtri=int(wm.sum()), near=near, dep=dep, dmax=round(float(dist_.max()), 1),
                                       dhist=np.bincount(np.minimum(dist_[dist_ > 0].astype(int), 30), minlength=31).tolist()))
                    if gd: gtrack[-1].update(ncl_old=ncl_old, f3=f3)
                    print("g", gtrack[-1]["t"], len(cl), near, gtrack[-1].get("ncl_old"), gtrack[-1].get("f3"), flush=True)
            if pf:
                if i % (5*ST) == 0 and not gz: snaps.append((t, psi.astype(np.float32), Phi.astype(np.float32), z.astype(np.complex64)))
                t2 = np.trace(Phi @ Phi, axis1=-2, axis2=-1).reshape(-1)
                rec.update(phiW=round(float(t2[B].mean()), 3) if B.any() else None, phiM=round(float(t2[~B].mean()), 4), wake=int((t2 > 0.3).sum()))
            if sn: snl.append((t, psi.astype(np.float32), z.astype(np.complex64), f.tail.astype(np.int32)))
            if snp and i % (snp*ST) == 0: spl.append((t, psi.astype(np.float32), z.astype(np.complex64)))   # No. 53: snapshots of ψ and z every snp units
            if sn and pf and gz: snv.append((np.linalg.eigvalsh(GF.P) > 0).sum(-1).astype(np.int8))   # (122): Φ★ vacuum at each node (2 — Φ★, 3 — −Φ★)   # (111): snapshots every unit to look for the birth "trigger"
            w2.append(rec)
    kick = None
    if Tk:                                                # (88): phase kick in the domain and in the Matrix (τ frozen), front of the winding difference
        B = psi > 0; Xc = f.xy[:, 0] + 0.5*f.xy[:, 1]; Yc = np.sqrt(3)/2*f.xy[:, 1]
        def bfs_(src, allow):
            dd = -np.ones(f.n, int); dd[src] = 0; q = list(src)
            while q:
                q2 = []
                for u_ in q:
                    for w_ in nb[u_]:
                        if allow[w_] and dd[w_] < 0: dd[w_] = dd[u_] + 1; q2.append(w_)
                q = q2
            return dd
        edge_ = [v for v in np.where(B)[0] if any(not B[w_] for w_ in nb[v])]
        dW = bfs_(edge_, B); cW = int(np.argmax(dW))                        # deepest node of the domain
        dM = bfs_(list(np.where(B)[0]), np.ones(f.n, bool)); cM = int(np.argmax(dM))   # Matrix node farthest from the domains
        def tdist(cx, cy, x, y):
            best = None
            for a in (-1, 0, 1):
                for b in (-1, 0, 1):
                    d_ = np.hypot(x - cx + a*L + b*L/2, y - cy + b*np.sqrt(3)/2*L); best = d_ if best is None else np.minimum(best, d_)
            return best
        dd_ = f.xy[f.head] - f.xy[f.tail]; dd_ = (dd_ + L//2) % L - L//2; mx = f.xy[f.tail] + dd_/2
        ex, ey = mx[:, 0] + 0.5*mx[:, 1], np.sqrt(3)/2*mx[:, 1]; TV = f.TV
        tx = (Xc[TV[:, 0]] + Xc[TV[:, 1]] + Xc[TV[:, 2]])/3; ty = (Yc[TV[:, 0]] + Yc[TV[:, 1]] + Yc[TV[:, 2]])/3
        bins = np.arange(0, 31, 1.0); out = dict(depthW=int(dW[cW]), distM=int(dM[cM]))
        cps = {"A": (z.copy(), None if wv is None else wv.copy(), psi.copy())}
        rts = {}
        for key, c in (("W", cW), ("M", cM)):
            re_ = tdist(Xc[c], Yc[c], ex, ey); rts[key] = tdist(Xc[c], Yc[c], tx, ty)
            kf = np.exp(1j*0.01*np.exp(-re_**2/(2*1.5**2))); cps[key] = (z*kf, None if wv is None else wv.copy(), psi.copy())
            out[key] = []
        tt = 0.0; j = 0
        while tt < Tk - 1e-9:
            for key in cps:
                z_, w_, p_ = cps[key]
                if mi: cps[key] = stepph(z_, w_, p_)
                else: z2, p2 = s.step(z_, p_, dt); cps[key] = (z2, None, p2)
            tt += dt; j += 1
            if j % max(1, int(round(0.25/dt))) == 0:
                l1, l2, l3 = f.tri_links(cps["A"][0]); PA = l1*l2*l3
                for key in ("W", "M"):
                    m1, m2, m3 = f.tri_links(cps[key][0]); dh = np.abs(np.angle((m1*m2*m3)/PA))
                    pr = np.histogram(rts[key], bins, weights=dh)[0]/np.maximum(np.histogram(rts[key], bins)[0], 1)
                    out[key].append([round(tt, 2)] + [float(x) for x in pr])
        kick = out
    B = psi > 0; final = BR.comps(f, B, nb) if B.any() else []
    np.savez(name + ".npz", z=z, psi=psi, bad=bad, m0=s.m0, **({"wv": wv} if mi else {}), **({"Phi": Phi} if pf else {}))
    if spl: np.savez(name + "_sp.npz", t=np.array([q[0] for q in spl]), psi=np.array([q[1] for q in spl]), z=np.array([q[2] for q in spl]))
    if sn: np.savez(name + "_sn.npz", t=np.array([q[0] for q in snl]), psi=np.array([q[1] for q in snl]), z=np.array([q[2] for q in snl]), tail=np.array([q[3] for q in snl]), **({"npos": np.array(snv)} if snv else {}))
    if pf: np.savez(name + "_g91.npz", psi=psi.reshape(L, L), P=Phi)
    if pf and snaps: np.savez(name + "_snap.npz", t=np.array([q[0] for q in snaps]), psi=np.array([q[1] for q in snaps]), P=np.array([q[2] for q in snaps]), z=np.array([q[3] for q in snaps]))   # (103)   # (92): input for gauge91 (ψ and Φ★ at the end of the run)
    if pf and gz: json.dump(dict(track=gtrack, plant=gplant), open(name + "_g104.json", "w")); np.savez(name + "_g104.npz", psi=psi, P=GF.P, U=GF.U)
    return dict(name=name, s0=s0, n=n, m0=s.m0, w2=w2, final=final[:30], kick=kick)

if __name__ == "__main__":
    kw = {k: (int(v) if k in ("L", "seed", "ln", "n", "k", "gseed", "cut", "ph", "vort", "iso", "pseed", "gz", "sn", "gk", "gd", "sk", "snp") else float(v)) for k, v in (a.split("=") for a in sys.argv[1:])}
    d = run(**kw); json.dump(d, open(d["name"] + ".json", "w"))
    t1 = next((q["t"] for q in d["w2"] if q["nuc"] > 0), None)
    print(d["name"], "m0", round(d["m0"], 3), "first nucleus t", t1, "final", d["final"][:12])
    for q in d["w2"][::10]: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in q.items()})
