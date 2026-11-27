"""runflip: модель local60 (рівняння без змін) на Матриці без розмірності (flipfab). Стріли — з τ самого графа (без нахилу і розпливання, як (66)).
Раз на одиницю часу: кипіння, вузли зі ψ > 0, світи (≥ 5 вузлів), середня кількість трикутників на вузол у світах і в нових вузлах світу
проти всієї Матриці. Наприкінці — знімок і ріст кулі всередині найбільшого світу (розмірність світу)."""
import os; os.environ["NB"] = "0"; os.environ.setdefault("DELTA", "1e-6")
import numpy as np, json, sys, scipy.sparse as sp
import flipfab as FF, simNlib as N, sim34d as S, local60 as LC, rnd51 as R, tile48 as TL, birth48 as BR

def run(L=96, q=0.0, gseed=0, seed=1, Tt=150.0, dt=0.005, n=100, ell=8.0, s0=3.5, h=0.3, lc=1.3, beta=1.5, eps0=3.0, kappa=1.0, ac=1.0):
    f = FF.FlipFabric(L, q, gseed); s = LC.Loc.__new__(LC.Loc); s.fab = f
    s.mod = S.M34(f, beta=beta, eps0=eps0, h=h, kappa=kappa); s.wp = N.wplus(beta); s.mod.wp = s.wp; s.lc = lc; s.cs, s.cc = 1.0 + 0j, 1 + 1j*ac
    tau = FF.tau_graph(f, ell, seed); bad = tau[f.head] - tau[f.tail] < 0; R.flip(f, bad)
    E3 = np.stack([f.E1, f.E2, f.E3], 1); rows = []; cols = []
    for a, b in ((0, 1), (1, 2), (0, 2)): rows += [E3[:, a], E3[:, b]]; cols += [E3[:, b], E3[:, a]]
    A = sp.csr_matrix((np.ones(sum(len(r) for r in rows)), (np.concatenate(rows), np.concatenate(cols))), shape=(f.m, f.m)); A.data[:] = 1.0
    s.P = ((sp.identity(f.m) + A)/5.0).tocsr(); assert np.allclose(np.asarray(s.P.sum(0)).ravel(), 1.0); s.n = n; s.tc = 1.0; s.K = None
    nb = [[] for _ in range(f.n)]
    for p, q_ in zip(f.tail, f.head): nb[p].append(q_); nb[q_].append(p)
    rng = np.random.default_rng(seed); r_ = np.roots([1, 0, -1, -h]); psiM = r_[np.isreal(r_)].real.min()
    z = np.sqrt(s0)*(1 + 0.01*(rng.standard_normal(f.m) + 1j*rng.standard_normal(f.m))); psi = psiM + 0.01*rng.standard_normal(f.n)
    s.S0 = float((np.abs(z)**2).sum()); s.m0 = s.S0/f.m
    name = f"rfl_L{L}_q{q:g}_g{gseed}_s{seed}_n{n}_ell{ell:g}" + (f"_e{eps0:g}" if eps0 != 3.0 else "") + (f"_b{beta:g}" if beta != 1.5 else "")
    t = 0.0; i = 0; ST = int(round(1/dt)); w2 = []; prev = None; Bp = np.zeros(f.n, bool); dg = f.deg/2.0          # трикутників на вузол = ступінь
    while t < Tt - 1e-9:
        z, psi = s.step(z, psi, dt); t += dt; i += 1
        if not np.isfinite(psi).all(): w2.append(dict(t=round(t, 2), blowup=True)); break
        if i % ST == 0:
            rec, prev = TL.stats(f, z, prev); B = psi > 0; cs = BR.comps(f, B, nb) if B.any() else []; new = B & ~Bp; Bp = B
            rec.update(t=round(t, 1), nB=int(B.sum()), nuc=sum(1 for c in cs if c >= 5), big=cs[0] if cs else 0, top=cs[:6],
                       degW=float(f.deg[B].mean()) if B.any() else None, degNew=float(f.deg[new].mean()) if new.any() else None, nNew=int(new.sum()))
            w2.append(rec); print(rec, flush=True)
    np.savez(name + ".npz", z=z, psi=psi, bad=bad, tau=tau, TV=f.TV)
    json.dump(dict(name=name, L=L, q=q, nflip=f.nflip, degmean=float(f.deg.mean()), degsd=float(f.deg.std()), w2=w2), open(name + ".json", "w"))
    return name

if __name__ == "__main__":
    kw = {k: (int(v) if k in ("L", "seed", "n", "gseed") else float(v)) for k, v in (a.split("=") for a in sys.argv[1:])}
    print(run(**kw))
