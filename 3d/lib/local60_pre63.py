"""local60: місцевий закон збереження натягу (розд. 4.4 (60), черга № 50). Як cons58 (kind=almost, L48, ψ вільне), але без глобального μ.
Модель сама створює/знищує натяг нитки зі швидкістю s_e = 2 Re(z̄ ż)_e. Місцевий закон: цей прибуток/збиток компенсується з околу нитки,
dρ_e/dt = s_e − (K s)_e, де K = P^n, P = (I + A)/5 (A — сусідство ниток через спільний трикутник; на правильній ґратці P двічі стохастична,
тож Σρ зберігається точно). Радіус околу ≈ √n ниток; n → ∞ — глобальний закон (58). Реалізація: ż_e −= (K s)_e/(2ρ_e)·z_e (ρ_e ≥ 0,05)."""
import numpy as np, sys, json, os
import scipy.sparse as sp
os.environ.setdefault("DELTA", "1e-6")
NBON = os.environ.get("NB", "1") == "1"                                   # компільована сила (nbforce), та сама формула
import nbforce as NB, fftk as FK, stall43 as T, rnd51 as R, tile48 as TL, birth48 as BR, seam51 as SM

class Loc(T.Sys):
    def F(s, z, psi, parts=False):
        if parts: return super().F(z, psi, parts)
        dz, dpsi = NB.force(s, z, psi) if NBON else super().F(z, psi); rho = np.maximum(np.abs(z)**2, 0.05)
        src = 2*np.real(np.conj(z)*dz)
        if getattr(s, 'K', None) is not None: ks = s.K(src)                 # P^n через FFT (fftk), точно те саме
        else:
            ks = src
            for _ in range(s.n): ks = s.P @ ks
        ks = ks + (rho.sum() - s.S0)/(s.tc*len(rho))                      # слабка поправка числового дрейфу
        return dz - ks/(2*rho)*z, dpsi

class LocR(Loc):                                                          # № 60: та сама модель на випадковій тріангуляції (rnd51)
    __init__ = R.Sys.__init__

def tau_field(L, ell, seed):
    """Плавне випадкове періодичне τ з масштабом ℓ (як tau52.tau_field, mode rand): гауссів фільтр білого шуму."""
    rng = np.random.default_rng(700 + seed); w = rng.standard_normal((L, L))
    kk = np.fft.fftfreq(L)*2*np.pi; K2 = kk[:, None]**2 + kk[None, :]**2
    t = np.real(np.fft.ifft2(np.fft.fft2(w)*np.exp(-K2*ell**2/2))); return t.ravel()/t.std()

def run(L=48, ln=46, s0=4.0, n=10, Tt=400.0, seed=1, h=0.3, dt=0.005, lc=1.3, tc=1.0, cont=0.0, k=1, ell=0.0, D=0.0, g=0.0, rnd=0.0, gseed=0):
    rng = np.random.default_rng(seed); r_ = np.roots([1, 0, -1, -h]); psiM = r_[np.isreal(r_)].real.min()
    s = LocR(L=L, gseed=gseed, jit=rnd, h=h, lc=lc) if rnd else Loc(L=L, h=h, lc=lc); f = s.fab
    if ell:                                             # № 42: стріли з випадкового плавного τ (як tau52), без швів від руки
        tau = tau_field(L, ell, seed)
        if rnd:                                          # № 60: τ у точці вузла — білінійна інтерполяція періодичного τ (без однакових τ у сусідів)
            T2 = tau.reshape(L, L); px, py = f.pos[:, 0] % L, f.pos[:, 1] % L; i0, j0 = np.floor(px).astype(int), np.floor(py).astype(int)
            u, v = px - i0, py - j0; i1, j1 = (i0 + 1) % L, (j0 + 1) % L
            tau = (1-u)*(1-v)*T2[i0, j0] + u*(1-v)*T2[i1, j0] + (1-u)*v*T2[i0, j1] + u*v*T2[i1, j1]
        #                                                № 58: τ = φ + g·(x + 0,31·y) — нахил без розриву (різниці по нитках, без обгортання тора)
        x_, y_ = f.xy[f.tail].T; x2, y2 = f.xy[f.head].T
        Gg = g*(((x2 - x_ + L//2) % L - L//2) + 0.31*((y2 - y_ + L//2) % L - L//2))    # внесок нахилу в Δτ нитки tail→head
        bad = tau[f.head] - tau[f.tail] + Gg < 0; R.flip(f, bad); Gg = np.where(bad, -Gg, Gg)
    else:
        R.flip(f, SM.seams(f))
        xa, ya = f.xy[f.tail].T; xb, yb = f.xy[f.head].T
        dy = (yb - ya + L//2) % L - L//2; dx = (xb - xa + L//2) % L - L//2; my = np.minimum(ya, yb)
        H = (dy == 0) & (np.abs(dx) == 1); V = (dx == 0) & (np.abs(dy) == 1); c = L//2
        cs_ = [(2*i + 1)*L//(2*k) for i in range(k)]      # k рядків і k стовпців швів: k*k хрестів (k=1 — один хрест, як раніше)
        bad = np.zeros(f.m, bool)
        for c in cs_: bad |= H & (ya == c) | V & (xa == c) & (my >= c - ln//2) & (my < c - ln//2 + ln)
        c = L//2; R.flip(f, bad)
    E3 = np.stack([f.E1, f.E2, f.E3], 1); rows = []; cols = []
    for a, b in ((0, 1), (1, 2), (0, 2)): rows += [E3[:, a], E3[:, b]]; cols += [E3[:, b], E3[:, a]]
    A = sp.csr_matrix((np.ones(sum(len(r) for r in rows)), (np.concatenate(rows), np.concatenate(cols))), shape=(f.m, f.m))
    A.data[:] = 1.0; s.P = ((sp.identity(f.m) + A)/5.0).tocsr(); s.n = n; s.tc = tc
    assert np.allclose(np.asarray(s.P.sum(0)).ravel(), 1.0)
    s.K = None if rnd else FK.build(f, L, s.P, n)          # FFT лише на правильній ґратці; на випадковій — множення на P
    nb = [[] for _ in range(f.n)]
    for p, q in zip(f.tail, f.head): nb[p].append(q); nb[q].append(p)
    if D: NBM = np.array([sorted(v) for v in nb])     # 6 сусідів кожного вузла (для лічби екстремумів τ)
    z = np.sqrt(s0)*(1 + 0.01*(rng.standard_normal(f.m) + 1j*rng.standard_normal(f.m))); psi = psiM + 0.01*rng.standard_normal(f.n)
    s.S0 = float((np.abs(z)**2).sum()); s.m0 = s.S0/f.m
    t = 0.0; i = 0; w2 = []; prev = None
    name = f"lc_s0{s0:g}_n{n}_L{L}_s{seed}" + (f"_k{k}" if k > 1 else "") + (f"_tau{ell:g}" if ell else "") + (f"_D{D:g}" if D else "") + (f"_g{g:g}" if g else "") + (f"_rnd{rnd:g}g{gseed}" if rnd else "")
    if cont:                                                              # продовження з t = cont: той самий стан, ті самі S0, P, дефекти
        d0 = np.load(name + ".npz"); assert (d0["bad"] == bad).all()
        z, psi = d0["z"], d0["psi"]; t = float(cont); i = int(round(cont/dt)); name += f"_T{Tt:g}"
    while t < Tt - 1e-9:
        z, psi = s.step(z, psi, dt); t += dt; i += 1
        if i % 200 == 0:
            if D:                                         # № 42 крок 2: τ розпливається (∂τ/∂t = D·Δτ на вузлах), стріли йдуть за τ
                ns = int(np.ceil(12*D))
                for _ in range(ns):
                    g_ = tau[f.head] - tau[f.tail]
                    tau = tau + (D*200*dt/ns)*(np.bincount(f.tail, g_, f.n) - np.bincount(f.head, g_, f.n))
                fl = tau[f.head] - tau[f.tail] + Gg < 0; R.flip(f, fl); z = np.where(fl, np.conj(z), z)   # стан нитки той самий, змінилась лише стріла
                Gg = np.where(fl, -Gg, Gg); nfl = int(fl.sum())
                od = np.bincount(f.tail, None, f.n); next_ = int(((od == 0) | (od == 6)).sum())         # екстремуми τ: усі стріли вузла всередину або назовні
                ndef = int((Gg < 0).sum()) if g else None                                              # нитки, що йдуть проти нахилу (неузгодженість відносно чистого нахилу)
            B = psi > 0; cs = BR.comps(f, B, nb) if B.any() else []
            Aa = np.abs(z)**2; w = (psi[f.tail] > 0) & (psi[f.head] > 0); o = (psi[f.tail] < 0) & (psi[f.head] < 0)
            rec, prev = TL.stats(f, z, prev)
            rec.update(t=round(t, 1), S=round(float(Aa.sum())/s.S0, 5), Ain=round(float(Aa[w].mean()), 3) if w.any() else None,
                       Aout=round(float(Aa[o].mean()), 3) if o.any() else None, nB=int(B.sum()), nuc=sum(1 for q_ in cs if q_ >= 5), big=cs[0] if cs else 0)
            if D: rec.update(nfl=nfl, next=next_, ndef=ndef)
            w2.append(rec)
    B = psi > 0; final = BR.comps(f, B, nb) if B.any() else []
    np.savez(name + ".npz", z=z, psi=psi, bad=bad, m0=s.m0)
    return dict(name=name, s0=s0, n=n, m0=s.m0, w2=w2, final=final[:30])

if __name__ == "__main__":
    kw = {k: (int(v) if k in ("L", "seed", "ln", "n", "k", "gseed") else float(v)) for k, v in (a.split("=") for a in sys.argv[1:])}
    d = run(**kw); json.dump(d, open(d["name"] + ".json", "w"))
    t1 = next((q["t"] for q in d["w2"] if q["nuc"] > 0), None)
    print(d["name"], "m0", round(d["m0"], 3), "first nucleus t", t1, "final", d["final"][:12])
    for q in d["w2"][::10]: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in q.items()})
