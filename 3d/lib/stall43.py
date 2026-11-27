"""stall43: чому бульбашка, народжена кипінням, зупиняється (розд. 4.4 (43)). Модель і зв'язок λ_c — як у nuc41.
mode=make: прогрів + народження + ріст до зупинки, зберегти стан (z, ψ, m0).
mode=diag: зі збереженого стану — середні за час складники сили на ψ і |z|² як функції відстані до межі.
mode=push: зі збереженого стану змінити один параметр (lc, eps0, h, kappa) і стежити за часткою бульбашки."""
import numpy as np, sys, json, os
os.environ.setdefault("DELTA", "1e-6")
import simNlib as N, sim34d as S
g = mu = 1.0

class Sys:
    def __init__(s, L=24, beta=1.5, eps0=3.0, h=0.3, kappa=1.0, as_=0.0, ac=1.0, lc=1.3, m0=-1.0):
        s.fab = N.Fabric(L); s.mod = S.M34(s.fab, beta=beta, eps0=eps0, h=h, kappa=kappa)
        s.wp = N.wplus(beta); s.lc, s.m0 = lc, (m0 if m0 > 0 else s.wp**2); s.cs, s.cc = 1 + 1j*as_, 1 + 1j*ac
    def F(s, z, psi, parts=False):
        f, mod, m = s.fab, s.mod, s.fab.m; a_, b_ = f.tail, f.head
        d = mod.rhs(0, mod.pack(z, psi)); dzf = d[:m] + 1j*d[m:2*m]; fpsi = d[2*m:].copy()
        sf = -(g + mu*np.abs(z)**2)*z; c = dzf - sf
        pe = (psi[a_] + psi[b_])/2
        sf = sf + s.lc*pe*z/(2*s.wp**2)
        q = s.lc*(np.abs(z)**2 - s.m0)/(4*s.wp**2)
        fc = np.bincount(a_, q, f.n) + np.bincount(b_, q, f.n)
        if parts:                                           # складники сили на ψ
            A = np.abs(z); we = np.sqrt(A**2 + mod.d2)/s.wp; dp = psi[b_] - psi[a_]
            grad = mod.kappa*(np.bincount(a_, we*dp, f.n) - np.bincount(b_, we*dp, f.n))
            pot = -(psi**3 - psi) + mod.h
            cost = fpsi - grad - pot                         # решта rhs ψ — ціна порушення ε₀
            return dict(pot=pot, grad=grad, cost=cost, coup=fc)
        return s.cs*sf + s.cc*c, fpsi + fc
    def step(s, z, psi, dt):
        k1 = s.F(z, psi); k2 = s.F(z + dt/2*k1[0], psi + dt/2*k1[1]); k3 = s.F(z + dt/2*k2[0], psi + dt/2*k2[1]); k4 = s.F(z + dt*k3[0], psi + dt*k3[1])
        return z + dt/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0]), psi + dt/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])

def make(L, lc, seed, T, h=0.3, warm=60.0, dt=0.005, **kw):
    s = Sys(L=L, lc=lc, h=h, **kw); fab = s.fab; rng = np.random.default_rng(seed); m = fab.m
    r = np.roots([1, 0, -1, -h]); r = r[np.isreal(r)].real; psiM = r.min()
    z = s.wp*(1 + 0.01*(rng.standard_normal(m) + 1j*rng.standard_normal(m))); psi = psiM + 0.01*rng.standard_normal(fab.n)
    acc = []; tw = 0.0; j = 0
    while tw < warm - 1e-9:
        k1 = s.F(z, psi)[0]; k2 = s.F(z + dt/2*k1, psi)[0]; k3 = s.F(z + dt/2*k2, psi)[0]; k4 = s.F(z + dt*k3, psi)[0]
        z = z + dt/6*(k1 + 2*k2 + 2*k3 + k4); tw += dt; j += 1
        if tw > warm/2 and j % 20 == 0: acc.append(float((np.abs(z)**2).mean()))
    s.m0 = float(np.mean(acc)); t = 0.0; i = 0; out = []
    while t < T - 1e-9:
        z, psi = s.step(z, psi, dt); t += dt; i += 1
        if i % 200 == 0: out.append((round(t, 1), float((psi > 0).mean())))
    np.savez(f"st_L{L}_l{lc}_s{seed}.npz", z=z, psi=psi, m0=s.m0, L=L, lc=lc, h=h)
    return dict(m0=s.m0, out=out[::10] + [out[-1]])

def dist(fab, B):
    """Відстань (у кроках графа) до межі: + у бульбашці, − у Матриці."""
    nb = [[] for _ in range(fab.n)]
    for a, b in zip(fab.tail, fab.head): nb[a].append(b); nb[b].append(a)
    D = np.full(fab.n, -1); fr = [v for v in range(fab.n) if any(B[w] != B[v] for w in nb[v])]
    for v in fr: D[v] = 0
    k = 0
    while fr:
        k += 1; nx = []
        for v in fr:
            for w in nb[v]:
                if D[w] < 0: D[w] = k; nx.append(w)
        fr = nx
    return np.where(B, D, -D) + np.where(B, 1, 0)           # межові вузли бульбашки = +1, Матриці = 0

def load(fn, **over):
    d = np.load(fn); p = dict(L=int(d['L']), lc=float(d['lc']), h=float(d['h']), m0=float(d['m0'])); p.update(over)
    return Sys(**p), d['z'], d['psi']

def diag(fn, T=60.0, dt=0.005):
    s, z, psi = load(fn); fab = s.fab; B0 = psi > 0; Dn = dist(fab, B0)
    acc = {k: np.zeros(fab.n) for k in ("pot", "grad", "cost", "coup", "A2", "psi")}; nrec = 0; t = 0.0; i = 0
    while t < T - 1e-9:
        z, psi = s.step(z, psi, dt); t += dt; i += 1
        if i % 20 == 0:
            P = s.F(z, psi, parts=True); A2 = np.abs(z)**2
            a2n = (np.bincount(fab.tail, A2, fab.n) + np.bincount(fab.head, A2, fab.n))/(np.bincount(fab.tail, None, fab.n) + np.bincount(fab.head, None, fab.n))
            for k in ("pot", "grad", "cost", "coup"): acc[k] += P[k]
            acc["A2"] += a2n; acc["psi"] += psi; nrec += 1
    prof = {}
    for dd in range(Dn.min(), Dn.max() + 1):
        sel = Dn == dd
        if sel.sum() < 3: continue
        prof[int(dd)] = {k: round(float(acc[k][sel].mean()/nrec), 4) for k in acc} | dict(n=int(sel.sum()))
    return dict(m0=s.m0, frac0=float(B0.mean()), frac1=float((psi > 0).mean()), prof=prof)

def push(fn, T=300.0, dt=0.005, **over):
    s, z, psi = load(fn)
    for k, v in over.items():
        if k == "lc": s.lc = v
        else: setattr(s.mod, k, v)
    t = 0.0; i = 0; out = [(0.0, float((psi > 0).mean()))]
    while t < T - 1e-9:
        z, psi = s.step(z, psi, dt); t += dt; i += 1
        if not np.isfinite(psi).all(): out.append((t, None)); break
        if i % 2000 == 0: out.append((round(t, 1), round(float((psi > 0).mean()), 4)))
    return dict(over=over, out=out)

if __name__ == "__main__":
    mode = sys.argv[1]; kw = dict(a.split("=") for a in sys.argv[2:])
    fn = kw.pop("fn", None); kw = {k: (int(v) if k in ("L", "seed") else float(v)) for k, v in kw.items()}
    if mode == "make": r = make(**kw)
    elif mode == "diag": r = diag(fn, **kw)
    else: r = push(fn, **kw)
    tag = "_".join(f"{k}{v}" for k, v in kw.items())
    json.dump(r, open(f"{mode}_{os.path.basename(fn or '')}_{tag}.json", "w")); print(json.dumps(r)[:3000])
