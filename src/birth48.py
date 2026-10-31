"""birth48: nucleation rate in the Matrix and whether a large Matrix boils (queue No. 35, sec. 4.4 (48)). Model — stall43 (λ_c = 1.3), torus.
Phase 1 (Tw): ψ frozen in the Matrix, we measure boiling: mean |z|², spread of |z|² over links, rate of change of |z|² (boil), phase disorder |hol|.
m0 = mean |z|² over the second half of phase 1 (self-consistent, as in (42)–(45)).
Phase 2 (Tb): ψ released, no planted bubble; every time unit — how many nodes are in the local phase and how many separate nuclei (≥ 5 nodes).
Nucleation rate = number of new nuclei / (number of nodes · time) up to the first large domain."""
import numpy as np, sys, json, os
os.environ.setdefault("DELTA", "1e-6")
import stall43 as T

def comps(fab, B, nb):
    seen = np.zeros(fab.n, bool); sizes = []
    for v in np.where(B)[0]:
        if seen[v]: continue
        st = [v]; seen[v] = True; c = 0
        while st:
            u = st.pop(); c += 1
            for w in nb[u]:
                if B[w] and not seen[w]: seen[w] = True; st.append(w)
        sizes.append(c)
    return sorted(sizes, reverse=True)

def run(L=96, lc=1.3, seed=0, h=0.3, dt=0.005, Tw=100.0, Tb=100.0):
    s = T.Sys(L=L, lc=lc, h=h); fab = s.fab; m = fab.m; rng = np.random.default_rng(seed)
    nb = [[] for _ in range(fab.n)]
    for a, b in zip(fab.tail, fab.head): nb[a].append(b); nb[b].append(a)
    r = np.roots([1, 0, -1, -h]); r = r[np.isreal(r)].real; psi = r.min() + 0.01*rng.standard_normal(fab.n)
    z = s.wp*(1 + 0.01*(rng.standard_normal(m) + 1j*rng.standard_normal(m)))
    t = 0.0; i = 0; w1 = []; prev = None
    while t < Tw - 1e-9:
        k1 = s.F(z, psi)[0]; k2 = s.F(z + dt/2*k1, psi)[0]; k3 = s.F(z + dt/2*k2, psi)[0]; k4 = s.F(z + dt*k3, psi)[0]
        z = z + dt/6*(k1 + 2*k2 + 2*k3 + k4); t += dt; i += 1
        if i % 200 == 0:
            A2 = np.abs(z)**2
            rec = dict(t=round(t, 1), A2=float(A2.mean()), sd=float(A2.std()), hol=float(np.abs(fab.holonomy(z)).mean()))
            if prev is not None: rec["boil"] = float(np.abs(A2 - prev).mean())
            prev = A2; w1.append(rec)
    half = [r for r in w1 if r["t"] > Tw/2]
    s.m0 = float(np.mean([r["A2"] for r in half]))
    t = 0.0; i = 0; w2 = []
    while t < Tb - 1e-9:
        z, psi = s.step(z, psi, dt); t += dt; i += 1
        if i % 200 == 0:
            B = psi > 0; c = comps(fab, B, nb) if B.any() else []
            w2.append(dict(t=round(t, 1), nB=int(B.sum()), nuc=sum(1 for x in c if x >= 5), big=c[0] if c else 0))
    summ = {k: float(np.mean([r[k] for r in half if k in r])) for k in ("A2", "sd", "boil", "hol")}
    return dict(L=L, lc=lc, seed=seed, n=fab.n, m0=s.m0, wp2=s.wp**2, boil=summ, w1=w1, w2=w2)

if __name__ == "__main__":
    kw = {k: (int(v) if k in ("L", "seed") else float(v)) for k, v in (a.split("=") for a in sys.argv[1:])}
    d = run(**kw); json.dump(d, open(f"birth_L{d['L']}_s{d['seed']}.json", "w"))
    print("L", d['L'], "m0", round(d['m0'], 3), "wp2", round(d['wp2'], 3), d['boil'])
    for r in d['w2'][::5]: print(r)
