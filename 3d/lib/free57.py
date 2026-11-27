"""free57: чи народжується світ без заморозки ψ (розд. 4.4 (57), черга № 47). Дефекти — як loop44 (kind=almost на L48, kind=open на L96).
ψ вільне від t = 0 (корінь Матриці + шум 0,01), кипіння і народження йдуть разом, T одиниць.
m0=fix:   m0 = число з попереднього прогону з заморозкою (m0fix).
m0=adapt: m0 підлаштовується на льоту: dm0/dt = (⟨|z|²⟩ − m0)/tau, середнє — по нитках вільної зони поза світами (ψ < 0 на обох кінцях); старт m0 = wp²."""
import numpy as np, sys, json, os
os.environ.setdefault("DELTA", "1e-6")
import stall43 as T, rnd51 as R, tile48 as TL, birth48 as BR, seam51 as SM, simNlib as N, loop44 as LP

def run(L=48, kind="almost", ln=46, rs=38, m0="adapt", m0fix=0.0, tau=10.0, Tt=250.0, seed=1, h=0.3, dt=0.005, lc=1.3, W=12.0):
    rng = np.random.default_rng(seed); r_ = np.roots([1, 0, -1, -h]); psiM = r_[np.isreal(r_)].real.min()
    s = LP.Open(L=L, h=h, lc=lc, W=W) if kind == "open" else T.Sys(L=L, h=h, lc=lc); f = s.fab; R.flip(f, SM.seams(f))
    xa, ya = f.xy[f.tail].T; xb, yb = f.xy[f.head].T
    dy = (yb - ya + L//2) % L - L//2; dx = (xb - xa + L//2) % L - L//2; mx = np.minimum(xa, xb); my = np.minimum(ya, yb)
    H = (dy == 0) & (np.abs(dx) == 1); V = (dx == 0) & (np.abs(dy) == 1); c = L//2
    if kind == "almost": bad = H & (ya == c) | V & (xa == c) & (my >= c - ln//2) & (my < c - ln//2 + ln)
    else: bad = H & (ya == c) & (np.abs(mx + 0.5 - c) < rs) | V & (xa == c) & (np.abs(my + 0.5 - c) < rs)
    R.flip(f, bad)
    free = s.r < s.Rb if kind == "open" else np.ones(f.n, bool); fe = free[f.tail] & free[f.head]
    nb = [[] for _ in range(f.n)]
    for p, q in zip(f.tail, f.head): nb[p].append(q); nb[q].append(p)
    s.m0 = m0fix if m0 == "fix" else s.wp**2
    z = s.wp*(1 + 0.01*(rng.standard_normal(f.m) + 1j*rng.standard_normal(f.m))); psi = psiM + 0.01*rng.standard_normal(f.n)
    t = 0.0; i = 0; w2 = []; prev = None
    while t < Tt - 1e-9:
        z, psi = s.step(z, psi, dt); t += dt; i += 1
        if m0 == "adapt" and i % 20 == 0:
            me = fe & (psi[f.tail] < 0) & (psi[f.head] < 0)
            s.m0 += 20*dt*(float((np.abs(z[me])**2).mean()) - s.m0)/tau
        if i % 200 == 0:
            B = (psi > 0) & free; cs = BR.comps(f, B, nb) if B.any() else []
            rec, prev = TL.stats(f, z, prev); rec.update(t=round(t, 1), m0=round(s.m0, 3), nB=int(B.sum()), nuc=sum(1 for q_ in cs if q_ >= 5), big=cs[0] if cs else 0)
            w2.append(rec)
    B = (psi > 0) & free; final = BR.comps(f, B, nb) if B.any() else []
    name = f"fr_{kind}_{m0}_L{L}_s{seed}"
    np.savez(name + ".npz", z=z, psi=psi, bad=bad, m0=s.m0)
    return dict(name=name, kind=kind, m0mode=m0, m0=s.m0, nbad=int(bad.sum()), w2=w2, final=final[:30])

if __name__ == "__main__":
    kw = {k: (v if k in ("kind", "m0") else int(v) if k in ("L", "seed", "ln", "rs") else float(v)) for k, v in (a.split("=") for a in sys.argv[1:])}
    d = run(**kw); json.dump(d, open(d["name"] + ".json", "w"))
    t1 = next((q["t"] for q in d["w2"] if q["nuc"] > 0), None)
    print(d["name"], "nbad", d["nbad"], "m0", round(d["m0"], 3), "first nucleus t", t1, "final", d["final"][:12])
    for q in d["w2"][::10]: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in q.items()})
