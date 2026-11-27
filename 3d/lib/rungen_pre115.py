"""rungen (114): модель local60 на 3D-сітках fabgen ('all' = FCC, 'planar', 'layer'). Нормування «на вузол, як у 2D» (варіант В):
β нитки = 2β/k_e (k_e — трикутників на нитку; вакуум кожної нитки той самий, що у 2D), ε₀ × 6/(трикутників на вузол), κ і λ_c × 6/(ниток на вузол).
Стріли — з плавного τ (ℓ = 3) + нахил g. Раз на одиницю часу: кипіння, світи, частка пласких вузлів (|гол.| < 0,35) — ознака протосвітів (111)–(113)."""
import os; os.environ["NB"] = "0"; os.environ.setdefault("DELTA", "1e-6")
import numpy as np, json, sys
import fabgen as G, simNlib as N, sim34d as S, local60 as LC, rnd51 as R, tile48 as TL, birth48 as BR
def run(L=16, kind="layer", seed=1, Tt=100.0, dt=0.005, n=100, ell=3.0, g=0.02, s0=3.5, h=0.3, lc=1.3, beta=1.5, eps0=3.0, kappa=1.0, ac=1.0, init="", pr=0.0):
    f = G.FabricG(L, kind); tn = float(f.tnode[0]); dg = float(f.deg[0])
    s = LC.Loc.__new__(LC.Loc); s.fab = f
    s.mod = S.M34(f, beta=2*beta/f.ktri, eps0=eps0*6/tn, h=h, kappa=kappa*6/dg); s.wp = N.wplus(beta); s.mod.wp = s.wp
    s.lc = lc*6/dg; s.cs, s.cc = 1.0 + 0j, 1 + 1j*ac
    tau = G.tau_cubic(L, ell, seed); Gg = g*(f.disp() @ np.array([1.0, 0.31, 0.17])); bad = tau[f.head] - tau[f.tail] + Gg < 0; R.flip(f, bad)
    P = G.local_P(f); s.P = P; s.n = n; s.tc = 1.0; s.K = G.fft_K(f, P, n)
    nb = [[] for _ in range(f.n)]
    for p, q in zip(f.tail, f.head): nb[p].append(q); nb[q].append(p)
    rng = np.random.default_rng(seed); r_ = np.roots([1, 0, -1, -h]); psiM = r_[np.isreal(r_)].real.min()
    z = np.sqrt(s0)*(1 + 0.01*(rng.standard_normal(f.m) + 1j*rng.standard_normal(f.m))); psi = psiM + 0.01*rng.standard_normal(f.n)
    if init:                                                             # (114): світ-куля радіуса pr (ψ = ψ світу) у центрі найбільшої пласкої ділянки знімка init
        d0 = np.load(init + ".npz"); assert (d0["bad"] == bad).all(); z = d0["z"]; psi = d0["psi"].copy()
        hol = np.abs(f.holonomy(z)); hn = np.bincount(f.TV.ravel(), np.repeat(hol, 3), f.n)/f.tnode; M = (hn < 0.35) & (psi < 0)
        cs_ = [];  seen = np.zeros(f.n, bool)
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
        dI = (f.ijk - f.ijk[c0] + L//2) % L - L//2; rr = np.linalg.norm(dI, axis=1); psi[rr <= pr] = r_[np.isreal(r_)].real.max()
        s.pzone = rr <= pr; print("plant at", f.ijk[c0], "patch", len(big_), "nodes", int(s.pzone.sum()), flush=True)
    s.S0 = float((np.abs(z)**2).sum()); s.m0 = s.S0/f.m
    name = f"rgen_{kind}_L{L}_s{seed}_n{n}" + (f"_ac{ac:g}" if ac != 1.0 else "") + (f"_e{eps0:g}" if eps0 != 3.0 else "") + (f"_pr{pr:g}" if init else "")
    t = 0.0; i = 0; ST = int(round(1/dt)); w2 = []; prev = None
    while t < Tt - 1e-9:
        z, psi = s.step(z, psi, dt); t += dt; i += 1
        if not np.isfinite(psi).all(): w2.append(dict(t=round(t, 2), blowup=True)); break
        if i % ST == 0:
            rec, prev = TL.stats(f, z, prev); B = psi > 0; cs = BR.comps(f, B, nb) if B.any() else []
            hol = np.abs(f.holonomy(z)); hn = np.bincount(f.TV.ravel(), np.repeat(hol, 3), f.n)/f.tnode
            A2n = (np.bincount(f.tail, np.abs(z)**2, f.n) + np.bincount(f.head, np.abs(z)**2, f.n))/f.deg
            rec.update(t=round(t, 1), nB=int(B.sum()), nuc=sum(1 for q in cs if q >= 5), big=cs[0] if cs else 0, top=cs[:6],
                       flat=float((hn < 0.35).mean()), proto=float(((hn < 0.35) & (A2n > 5)).mean()), holm=float(hn.mean()))
            w2.append(rec); print(rec, flush=True)
    np.savez(name + ".npz", z=z, psi=psi, bad=bad)
    json.dump(dict(name=name, kind=kind, L=L, ntri=f.ntri, m=f.m, w2=w2), open(name + ".json", "w")); return name
if __name__ == "__main__":
    kw = {k: (v if k in ("kind", "init") else (int(v) if k in ("L", "seed", "n") else float(v))) for k, v in (a.split("=") for a in sys.argv[1:])}
    print(run(**kw))
