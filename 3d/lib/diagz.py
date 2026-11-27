"""diagz (118): чому нитки під посадженим світом у 3D не слабнуть. Розклад d|z|²/dt ниток усередині світу на складники:
self −2(g+μ|z|²)|z|², coup (λ_c·ψ), beta (клітинки, без повороту), beta_rot (поворот α клітинок), eps (порушення ε₀), eps_rot, kap (κ), loc (місцевий закон).
2D: посадка як plant2d (знімок L96, R = 6). Б: посадка як rungen4 (тетраедри, R = 3,5 або 4,5)."""
import os; os.environ["NB"] = "0"; os.environ.setdefault("DELTA", "1e-6")
import numpy as np, json, sys
import local60 as LC
g = mu = 1.0
def zparts(s, z, psi, inn):
    mod = s.mod; m = s.fab.m; e0, k0 = mod.eps0, mod.kappa
    def c(eps, kap):
        mod.eps0, mod.kappa = eps, kap; d = mod.rhs(0, mod.pack(z, psi)); return d[:m] + 1j*d[m:2*m]
    call = c(e0, k0); cnoE = c(0.0, k0); cnoEK = c(0.0, 0.0); mod.eps0, mod.kappa = e0, k0
    sf0 = -(g + mu*np.abs(z)**2)*z; a_, b_ = s.fab.tail, s.fab.head
    coup = s.lc*(psi[a_] + psi[b_])/2*z/(2*s.wp**2)
    beta, eps, kap = cnoEK - sf0, call - cnoE, cnoE - cnoEK
    full = s.F(z, psi)[0]; base = LC.T.Sys.F(s, z, psi)[0]; loc = full - base
    al = s.cc.imag; R_ = lambda X: float((2*np.real(np.conj(z)*X))[inn].mean()); I_ = lambda X: float((-2*al*np.imag(np.conj(z)*X))[inn].mean())
    return dict(A2=float((np.abs(z)**2)[inn].mean()), self=R_(sf0), coup=R_(coup), beta=R_(beta), beta_rot=I_(beta), eps=R_(eps), eps_rot=I_(eps),
                kap=R_(kap), kap_rot=I_(kap), loc=R_(loc), total=R_(full))
def evolve(s, z, psi, inn, zone, Tt=1.5, dt=0.005, every=0.1):
    out = []; t = 0.0; k = 0; E = int(round(every/dt))
    while t < Tt + 1e-9:
        if k % E == 0: r = zparts(s, z, psi, inn); r.update(t=round(t, 2), psi=float(psi[zone].mean())); out.append(r); print({a: round(b, 3) for a, b in r.items()}, flush=True)
        z, psi = s.step(z, psi, dt); t += dt; k += 1
    return out
def run2d(pr=6.0):
    import scipy.sparse as sp, rnd51 as R, fftk as FK
    nm = "lc_s03.5_n100_L96_s1_tau8_D1_g0.02_iso"; d = np.load(nm + ".npz"); L = 96; s = LC.Loc(L=L, h=0.3, lc=1.3); f = s.fab; R.flip(f, d["bad"])
    E3 = np.stack([f.E1, f.E2, f.E3], 1); rows = []; cols = []
    for a, b in ((0, 1), (1, 2), (0, 2)): rows += [E3[:, a], E3[:, b]]; cols += [E3[:, b], E3[:, a]]
    A = sp.csr_matrix((np.ones(sum(len(r) for r in rows)), (np.concatenate(rows), np.concatenate(cols))), shape=(f.m, f.m)); A.data[:] = 1.0
    s.P = ((sp.identity(f.m) + A)/5.0).tocsr(); s.n = 100; s.tc = 1.0; s.K = FK.build(f, L, s.P, 100)
    z = d["z"]; psi = d["psi"].copy(); s.S0 = float((np.abs(z)**2).sum()); s.m0 = float(d["m0"])
    from scipy.ndimage import distance_transform_edt as edt
    W = (psi > 0).reshape(L, L); far = edt(~np.tile(W, (3, 3)))[L:2*L, L:2*L]; c = np.unravel_index(far.argmax(), far.shape)
    i, j = f.xy.T; di = (i - c[0] + L//2) % L - L//2; dj = (j - c[1] + L//2) % L - L//2
    rr = np.hypot(di + dj/2, dj*np.sqrt(3)/2); r_ = np.roots([1, 0, -1, -0.3]); psi[rr <= pr] = r_[np.isreal(r_)].real.max()
    zone = rr <= pr; inn = zone[f.tail] & zone[f.head]; return evolve(s, z, psi, inn, zone)
def run3d(seed=1, pr=3.5, eps0=3.0):
    import rungen4 as Q
    s, f, mod, z, psi = Q.run(L=16, seed=seed, check=1, eps0=eps0)
    nm = f"rq_L16_s{seed}_n100"; d0 = np.load(nm + ".npz"); z = d0["z"]; psi = d0["psi"].copy(); L = 16
    hol = np.abs(f.holonomy(z)); hn = np.bincount(f.TV.ravel(), np.repeat(hol, 3), f.n)/f.tnode; M = (hn < 0.35) & (psi < 0)
    nb = [[] for _ in range(f.n)]
    for p, q in zip(f.tail, f.head): nb[p].append(q); nb[q].append(p)
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
    dI = (f.ijk - f.ijk[c0] + L//2) % L - L//2; rr = np.linalg.norm(dI, axis=1); r_ = np.roots([1, 0, -1, -0.3]); psi[rr <= pr] = r_[np.isreal(r_)].real.max()
    s.S0 = float((np.abs(z)**2).sum()); s.m0 = s.S0/f.m
    zone = rr <= pr; inn = zone[f.tail] & zone[f.head]; return evolve(s, z, psi, inn, zone)
if __name__ == "__main__":
    which = sys.argv[1]; kw = {k: float(v) if k != "seed" else int(v) for k, v in (a.split("=") for a in sys.argv[2:])}
    out = run2d(**kw) if which == "2d" else run3d(**kw)
    json.dump(out, open(f"diagz_{which}_" + "_".join(f"{k}{v}" for k, v in kw.items()) + ".json", "w"))
