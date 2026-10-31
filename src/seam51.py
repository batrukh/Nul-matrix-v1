"""seam51: is the boiling of the Matrix on the regular lattice seeded by "seams" of link directions (sec. 4.4 (51)).
In simNlib.Fabric the link direction is set by the creation order: almost all links point the same way, except row 0 (horizontal)
and column 0 (vertical), where the direction is opposite. With a rotated force (α ≠ 0) the direction is physical, so the seams are defects.
mode=fresh: regular lattice without seams (seams flipped), start from a nearly quiet Matrix, ψ frozen, T units.
mode=keep:  the same with seams (control).
mode=heal:  first T/2 with seams (the Matrix boils), then the seams are flipped (z → z̄ on them, the same physical state) and another T/2."""
import numpy as np, sys, json, os
os.environ.setdefault("DELTA", "1e-6")
import stall43 as T, rnd51 as R, tile48 as TL

def seams(fab):
    L = fab.L; xa, ya = fab.xy[fab.tail].T; xb, yb = fab.xy[fab.head].T
    dx = (xb - xa + L//2) % L - L//2; dy = (yb - ya + L//2) % L - L//2
    return (dx == 1) & (dy == 0) | (dx == 0) & (dy == -1)        # against the direction of the other links of its family

def run(mode="fresh", L=48, seed=1, Tt=200.0, dt=0.005, lc=1.3, h=0.3, noise=0.01):
    rng = np.random.default_rng(seed); r = np.roots([1, 0, -1, -h]); psiM = r[np.isreal(r)].real.min()
    s = T.Sys(L=L, lc=lc, h=h); f = s.fab; sm = seams(f); psi = np.full(f.n, psiM)
    z = s.wp*(1 + 0.01*(rng.standard_normal(f.m) + 1j*rng.standard_normal(f.m))); log = []
    if mode == "fresh": R.flip(f, sm); z = s.wp*(1 + noise*(rng.standard_normal(f.m) + 1j*rng.standard_normal(f.m)))
    if mode == "heal":
        z = TL.evolve(s, z, psi, Tt/2, dt, log); R.flip(f, sm); z = np.where(sm, np.conj(z), z)
        l2 = []; z = TL.evolve(s, z, psi, Tt/2, dt, l2); log += [dict(q, t=q["t"] + Tt/2) for q in l2]
    else: z = TL.evolve(s, z, psi, Tt, dt, log)
    return dict(mode=mode, L=L, seed=seed, nseam=int(sm.sum()), m=f.m, log=log)

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[1:]); kw = {k: (v if k == "mode" else int(v) if k in ("L", "seed") else float(v)) for k, v in kw.items()}
    d = run(**kw); json.dump(d, open(f"seam_{d['mode']}_L{d['L']}_s{d['seed']}_T{int(kw.get('Tt', 200))}_n{kw.get('noise', 0.01)}.json", "w"))
    print(d['mode'], d['L'], d['nseam'], d['m'])
    for q in d['log'][::10] + [d['log'][-1]]: print({k: round(v, 3) for k, v in q.items()})
