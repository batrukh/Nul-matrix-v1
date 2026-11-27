"""local60: місцевий закон збереження натягу (розд. 4.4 (60), черга № 50). Як cons58 (kind=almost, L48, ψ вільне), але без глобального μ.
Модель сама створює/знищує натяг нитки зі швидкістю s_e = 2 Re(z̄ ż)_e. Місцевий закон: цей прибуток/збиток компенсується з околу нитки,
dρ_e/dt = s_e − (K s)_e, де K = P^n, P = (I + A)/5 (A — сусідство ниток через спільний трикутник; на правильній ґратці P двічі стохастична,
тож Σρ зберігається точно). Радіус околу ≈ √n ниток; n → ∞ — глобальний закон (58). Реалізація: ż_e −= (K s)_e/(2ρ_e)·z_e (ρ_e ≥ 0,05)."""
import numpy as np, sys, json, os
import scipy.sparse as sp
os.environ.setdefault("DELTA", "1e-6")
import fftk as FK, stall43 as T, rnd51 as R, tile48 as TL, birth48 as BR, seam51 as SM

class Loc(T.Sys):
    def F(s, z, psi, parts=False):
        if parts: return super().F(z, psi, parts)
        dz, dpsi = super().F(z, psi); rho = np.maximum(np.abs(z)**2, 0.05)
        src = 2*np.real(np.conj(z)*dz)
        if getattr(s, 'K', None) is not None: ks = s.K(src)                 # P^n через FFT (fftk), точно те саме
        else:
            ks = src
            for _ in range(s.n): ks = s.P @ ks
        ks = ks + (rho.sum() - s.S0)/(s.tc*len(rho))                      # слабка поправка числового дрейфу
        return dz - ks/(2*rho)*z, dpsi

def run(L=48, ln=46, s0=4.0, n=10, Tt=400.0, seed=1, h=0.3, dt=0.005, lc=1.3, tc=1.0, cont=0.0):
    rng = np.random.default_rng(seed); r_ = np.roots([1, 0, -1, -h]); psiM = r_[np.isreal(r_)].real.min()
    s = Loc(L=L, h=h, lc=lc); f = s.fab; R.flip(f, SM.seams(f))
    xa, ya = f.xy[f.tail].T; xb, yb = f.xy[f.head].T
    dy = (yb - ya + L//2) % L - L//2; dx = (xb - xa + L//2) % L - L//2; my = np.minimum(ya, yb)
    H = (dy == 0) & (np.abs(dx) == 1); V = (dx == 0) & (np.abs(dy) == 1); c = L//2
    bad = H & (ya == c) | V & (xa == c) & (my >= c - ln//2) & (my < c - ln//2 + ln); R.flip(f, bad)
    E3 = np.stack([f.E1, f.E2, f.E3], 1); rows = []; cols = []
    for a, b in ((0, 1), (1, 2), (0, 2)): rows += [E3[:, a], E3[:, b]]; cols += [E3[:, b], E3[:, a]]
    A = sp.csr_matrix((np.ones(sum(len(r) for r in rows)), (np.concatenate(rows), np.concatenate(cols))), shape=(f.m, f.m))
    A.data[:] = 1.0; s.P = ((sp.identity(f.m) + A)/5.0).tocsr(); s.n = n; s.tc = tc
    assert np.allclose(np.asarray(s.P.sum(0)).ravel(), 1.0)
    s.K = FK.build(f, L, s.P, n)
    nb = [[] for _ in range(f.n)]
    for p, q in zip(f.tail, f.head): nb[p].append(q); nb[q].append(p)
    z = np.sqrt(s0)*(1 + 0.01*(rng.standard_normal(f.m) + 1j*rng.standard_normal(f.m))); psi = psiM + 0.01*rng.standard_normal(f.n)
    s.S0 = float((np.abs(z)**2).sum()); s.m0 = s.S0/f.m
    t = 0.0; i = 0; w2 = []; prev = None
    name = f"lc_s0{s0:g}_n{n}_L{L}_s{seed}"
    if cont:                                                              # продовження з t = cont: той самий стан, ті самі S0, P, дефекти
        d0 = np.load(name + ".npz"); assert (d0["bad"] == bad).all()
        z, psi = d0["z"], d0["psi"]; t = float(cont); i = int(round(cont/dt)); name += f"_T{Tt:g}"
    while t < Tt - 1e-9:
        z, psi = s.step(z, psi, dt); t += dt; i += 1
        if i % 200 == 0:
            B = psi > 0; cs = BR.comps(f, B, nb) if B.any() else []
            Aa = np.abs(z)**2; w = (psi[f.tail] > 0) & (psi[f.head] > 0); o = (psi[f.tail] < 0) & (psi[f.head] < 0)
            rec, prev = TL.stats(f, z, prev)
            rec.update(t=round(t, 1), S=round(float(Aa.sum())/s.S0, 5), Ain=round(float(Aa[w].mean()), 3) if w.any() else None,
                       Aout=round(float(Aa[o].mean()), 3) if o.any() else None, nB=int(B.sum()), nuc=sum(1 for q_ in cs if q_ >= 5), big=cs[0] if cs else 0)
            w2.append(rec)
    B = psi > 0; final = BR.comps(f, B, nb) if B.any() else []
    np.savez(name + ".npz", z=z, psi=psi, bad=bad, m0=s.m0)
    return dict(name=name, s0=s0, n=n, m0=s.m0, w2=w2, final=final[:30])

if __name__ == "__main__":
    kw = {k: (int(v) if k in ("L", "seed", "ln", "n") else float(v)) for k, v in (a.split("=") for a in sys.argv[1:])}
    d = run(**kw); json.dump(d, open(d["name"] + ".json", "w"))
    t1 = next((q["t"] for q in d["w2"] if q["nuc"] > 0), None)
    print(d["name"], "m0", round(d["m0"], 3), "first nucleus t", t1, "final", d["final"][:12])
    for q in d["w2"][::10]: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in q.items()})
