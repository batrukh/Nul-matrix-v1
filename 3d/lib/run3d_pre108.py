"""run3d: модель local60 на 3D-Матриці (fab3d, FCC), черга № 60 крок 1. Рівняння ті самі (sim34d.M34 + stall43.Sys + місцевий закон local60).
Стріли — з нерухомого до-часу τ = φ (плавне випадкове, ℓ) + нахил g·(d·w), w = (1; 0,31; 0,17) (як (66)+(68), без розпливання).
Варіанти сталих: var=A — β, ε₀, κ, λ_c як у 2D; запас інтенсивності s0 у тій самій частці від вакууму w₊², бо в 3D нитка в 4 трикутниках
(вакуум μa² − 4βa + g = 0 → a = w₊(2β)). var=B — нормування на нитку: β, ε₀, κ, λ_c удвічі менші, вакуум і запас як у 2D.
Міряємо раз на одиницю часу: середнє і розкид |z|², кипіння (зміна |z|²), частку ψ > 0, зародки (≥ 5 вузлів), найбільший світ."""
import os; os.environ["NB"] = "0"; os.environ.setdefault("DELTA", "1e-6")
import numpy as np, json, sys
import fab3d as F, simNlib as N, sim34d as S, local60 as LC, rnd51 as R, tile48 as TL, birth48 as BR

def run(L=16, var="A", seed=1, Tt=150.0, dt=0.005, n=100, ell=3.0, g=0.02, s0f=3.5/6.8541, h=0.3, lc=1.3, beta=1.5, eps0=3.0, kappa=1.0, ac=1.0):
    f = F.Fabric3D(L); fac = 1.0 if var == "A" else 0.5
    s = LC.Loc.__new__(LC.Loc); s.fab = f
    s.mod = S.M34(f, beta=beta*fac, eps0=eps0*fac, h=h, kappa=kappa*fac)
    s.wp = N.wplus(2*beta*fac); s.mod.wp = s.wp; s.lc = lc*fac; s.cs, s.cc = 1.0 + 0j, 1 + 1j*ac
    tau = F.tau3d(L, ell, seed); d = f.disp(); Gg = g*(d @ np.array([1.0, 0.31, 0.17]))
    bad = tau[f.head] - tau[f.tail] + Gg < 0; R.flip(f, bad)
    P = F.local_P(f); s.P = P; s.n = n; s.tc = 1.0; s.K = F.fft_K(f, P, n)
    nb = [[] for _ in range(f.n)]
    for p, q in zip(f.tail, f.head): nb[p].append(q); nb[q].append(p)
    rng = np.random.default_rng(seed); r_ = np.roots([1, 0, -1, -h]); psiM = r_[np.isreal(r_)].real.min()
    s0 = s0f*s.wp**2
    z = np.sqrt(s0)*(1 + 0.01*(rng.standard_normal(f.m) + 1j*rng.standard_normal(f.m))); psi = psiM + 0.01*rng.standard_normal(f.n)
    s.S0 = float((np.abs(z)**2).sum()); s.m0 = s.S0/f.m
    name = f"r3d_L{L}_{var}_s{seed}_n{n}_ell{ell:g}_g{g:g}" + (f"_e{eps0:g}" if eps0 != 3.0 else "") + (f"_h{h:g}" if h != 0.3 else "")
    t = 0.0; i = 0; ST = int(round(1/dt)); w2 = []; prev = None
    while t < Tt - 1e-9:
        z, psi = s.step(z, psi, dt); t += dt; i += 1
        if not np.isfinite(psi).all(): w2.append(dict(t=round(t, 2), blowup=True)); break
        if i % ST == 0:
            rec, prev = TL.stats(f, z, prev); B = psi > 0; cs = BR.comps(f, B, nb) if B.any() else []
            rec.update(t=round(t, 1), wp2=round(s.wp**2, 3), S=round(float((np.abs(z)**2).sum())/s.S0, 5), nB=int(B.sum()),
                       nuc=sum(1 for q in cs if q >= 5), big=cs[0] if cs else 0, top=cs[:6])
            w2.append(rec); print(rec, flush=True)
    np.savez(name + ".npz", z=z, psi=psi, bad=bad)
    json.dump(dict(name=name, var=var, L=L, n=n, s0=s0, wp2=s.wp**2, ndef=int(bad.sum()), w2=w2), open(name + ".json", "w"))
    return name

if __name__ == "__main__":
    kw = {k: (v if k == "var" else (int(v) if k in ("L", "seed", "n") else float(v))) for k, v in (a.split("=") for a in sys.argv[1:])}
    print(run(**kw))
