"""tile48: does a large Matrix sustain boiling (queue No. 35, sec. 4.4 (48)). We warm up a small Matrix Ls (ψ frozen) to steady boiling,
tile it onto a large lattice L = k·Ls, add noise (breaks the periodicity) and check whether boiling persists (|z|² stays low)
or dies out to rest (|z|² → w₊² ≈ 6.85), as it would if boiling were an effect of the small torus."""
import numpy as np, sys, json
import stall43 as T

def stats(fab, z, prev):
    A2 = np.abs(z)**2
    rec = dict(A2=float(A2.mean()), sd=float(A2.std()), hol=float(np.abs(fab.holonomy(z)).mean()))
    if prev is not None: rec["boil"] = float(np.abs(A2 - prev).mean())
    return rec, A2

def evolve(s, z, psi, Tt, dt, log):
    t = 0.0; i = 0; prev = None
    while t < Tt - 1e-9:
        k1 = s.F(z, psi)[0]; k2 = s.F(z + dt/2*k1, psi)[0]; k3 = s.F(z + dt/2*k2, psi)[0]; k4 = s.F(z + dt*k3, psi)[0]
        z = z + dt/6*(k1 + 2*k2 + 2*k3 + k4); t += dt; i += 1
        if i % 200 == 0:
            rec, prev = stats(s.fab, z, prev); rec["t"] = round(t, 1); log.append(rec)
    return z

def run(Ls=48, k=3, seed=0, Ts=100.0, Tb=100.0, noise=0.05, h=0.3, dt=0.005):
    rng = np.random.default_rng(seed); r = np.roots([1, 0, -1, -h]); psiM = r[np.isreal(r)].real.min()
    s = T.Sys(L=Ls, h=h); fs = s.fab
    z = s.wp*(1 + 0.01*(rng.standard_normal(fs.m) + 1j*rng.standard_normal(fs.m))); psi = np.full(fs.n, psiM)
    logS = []; z = evolve(s, z, psi, Ts, dt, logS)
    L = k*Ls; b = T.Sys(L=L, h=h); fb = b.fab
    key = {(min(a, c), max(a, c)): (e, a) for e, (a, c) in enumerate(zip(fs.tail, fs.head))}
    sm = lambda v: (fb.xy[v, 0] % Ls)*Ls + (fb.xy[v, 1] % Ls)
    zb = np.empty(fb.m, complex)
    for e, (a, c) in enumerate(zip(fb.tail, fb.head)):
        us, vs = sm(a), sm(c); e2, t2 = key[(min(us, vs), max(us, vs))]
        zb[e] = z[e2] if t2 == us else np.conj(z[e2])
    zb = zb*(1 + noise*(rng.standard_normal(fb.m) + 1j*rng.standard_normal(fb.m)))
    logB = []; evolve(b, zb, np.full(fb.n, psiM), Tb, dt, logB)
    return dict(Ls=Ls, L=L, seed=seed, noise=noise, small=logS, big=logB)

if __name__ == "__main__":
    kw = {kk: (int(v) if kk in ("Ls", "k", "seed") else float(v)) for kk, v in (a.split("=") for a in sys.argv[1:])}
    d = run(**kw); json.dump(d, open(f"tile_{d['Ls']}x{kw.get('k', 3)}_s{d['seed']}.json", "w"))
    print("small end", {kk: round(v, 3) for kk, v in d['small'][-1].items()})
    for r in d['big'][::10] + [d['big'][-1]]: print({kk: round(v, 3) for kk, v in r.items()})
