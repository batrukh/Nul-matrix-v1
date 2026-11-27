"""cons58: збережений натяг (журнал E, 17.10.2026; розд. 4.4 (58), черга № 48). ψ вільне від t = 0, як free57.
Закон збереження: S = Σ|z|² по нитках E (уся Матриця; на відкритій — вільна зона) стале. До сили на нитки додано −μ(t)·z,
μ = Re Σ z̄·ż / S (+ слабка поправка дрейфу (S − S0)/(S0·tc)), тобто ніщо не створює і не знищує натяг, лише перерозподіляє.
m0 = S0/|E| — сталий середній натяг Матриці (не ручне число). S0 = натяг на старті (wp² на нитку) або s0 на нитку.
kind=almost (L48), open (L96), none (L48, без дефектів — контроль)."""
import numpy as np, sys, json, os
os.environ.setdefault("DELTA", "1e-6")
import stall43 as T, rnd51 as R, tile48 as TL, birth48 as BR, seam51 as SM, simNlib as N, loop44 as LP

def cons(base):
    class C(base):
        def F(s, z, psi, parts=False):
            if parts: return super().F(z, psi, parts)
            dz, dpsi = super().F(z, psi); E = s.E; zE = z[E]; S = float((np.abs(zE)**2).sum())
            mu = float(np.real(np.vdot(zE, dz[E])))/S + (S - s.S0)/(s.S0*s.tc)
            dz = dz.copy(); dz[E] -= mu*zE
            return dz, dpsi
    return C

def run(L=48, kind="almost", ln=46, rs=38, s0=0.0, Tt=250.0, seed=1, h=0.3, dt=0.005, lc=1.3, W=12.0, tc=1.0):
    rng = np.random.default_rng(seed); r_ = np.roots([1, 0, -1, -h]); psiM = r_[np.isreal(r_)].real.min()
    s = cons(LP.Open)(L=L, h=h, lc=lc, W=W) if kind == "open" else cons(T.Sys)(L=L, h=h, lc=lc); f = s.fab; R.flip(f, SM.seams(f))
    xa, ya = f.xy[f.tail].T; xb, yb = f.xy[f.head].T
    dy = (yb - ya + L//2) % L - L//2; dx = (xb - xa + L//2) % L - L//2; mx = np.minimum(xa, xb); my = np.minimum(ya, yb)
    H = (dy == 0) & (np.abs(dx) == 1); V = (dx == 0) & (np.abs(dy) == 1); c = L//2
    if kind == "almost": bad = H & (ya == c) | V & (xa == c) & (my >= c - ln//2) & (my < c - ln//2 + ln)
    elif kind == "open": bad = H & (ya == c) & (np.abs(mx + 0.5 - c) < rs) | V & (xa == c) & (np.abs(my + 0.5 - c) < rs)
    else: bad = np.zeros(f.m, bool)
    R.flip(f, bad)
    free = s.r < s.Rb if kind == "open" else np.ones(f.n, bool); fe = free[f.tail] & free[f.head]
    nb = [[] for _ in range(f.n)]
    for p, q in zip(f.tail, f.head): nb[p].append(q); nb[q].append(p)
    z = s.wp*(1 + 0.01*(rng.standard_normal(f.m) + 1j*rng.standard_normal(f.m))); psi = psiM + 0.01*rng.standard_normal(f.n)
    if s0 > 0: z = z*np.sqrt(s0)/s.wp
    s.E = np.flatnonzero(fe); s.tc = tc; s.S0 = float((np.abs(z[s.E])**2).sum()); s.m0 = s.S0/len(s.E)
    t = 0.0; i = 0; w2 = []; prev = None
    while t < Tt - 1e-9:
        z, psi = s.step(z, psi, dt); t += dt; i += 1
        if i % 200 == 0:
            B = (psi > 0) & free; cs = BR.comps(f, B, nb) if B.any() else []
            A = np.abs(z)**2; w = fe & (psi[f.tail] > 0) & (psi[f.head] > 0); o = fe & (psi[f.tail] < 0) & (psi[f.head] < 0)
            rec, prev = TL.stats(f, z, prev)
            rec.update(t=round(t, 1), S=round(float(A[fe].sum())/s.S0, 5), Ain=round(float(A[w].mean()), 3) if w.any() else None,
                       Aout=round(float(A[o].mean()), 3) if o.any() else None, nB=int(B.sum()), nuc=sum(1 for q_ in cs if q_ >= 5), big=cs[0] if cs else 0)
            w2.append(rec)
    B = (psi > 0) & free; final = BR.comps(f, B, nb) if B.any() else []
    name = f"cs_{kind}_s0{s0:g}_L{L}_s{seed}"
    np.savez(name + ".npz", z=z, psi=psi, bad=bad, m0=s.m0)
    return dict(name=name, kind=kind, m0=s.m0, nfree=int(free.sum()), nbad=int(bad.sum()), w2=w2, final=final[:30])

if __name__ == "__main__":
    kw = {k: (v if k == "kind" else int(v) if k in ("L", "seed", "ln", "rs") else float(v)) for k, v in (a.split("=") for a in sys.argv[1:])}
    d = run(**kw); json.dump(d, open(d["name"] + ".json", "w"))
    t1 = next((q["t"] for q in d["w2"] if q["nuc"] > 0), None)
    print(d["name"], "nfree", d["nfree"], "m0", round(d["m0"], 3), "first nucleus t", t1, "final", d["final"][:12])
    for q in d["w2"][::10]: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in q.items()})
