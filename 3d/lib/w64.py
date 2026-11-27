"""w64: № 64 крок 1 — однорідний стан світу (ψ > 0 усюди) з місцевим законом і стрілами чистого нахилу τ = g·(i + 0,31 j).
Чи такий стан стійкий (не розпадається, не кипить), яка інтенсивність ниток за напрямками. Контроль — α = 0 (стріли не діють)."""
import numpy as np, sys, json
import scipy.sparse as sp
import local60 as M, rnd51 as R, fftk as FK, tile48 as TL
def build(L=48, s0=6.0, ac=1.0, n=100, seed=1, tilt=1, h=0.3, lc=1.3):
    s = M.Loc(L=L, h=h, lc=lc, ac=ac); f = s.fab
    if tilt:
        d = f.xy[f.head] - f.xy[f.tail]; d = (d + L//2) % L - L//2
        R.flip(f, d[:, 0] + 0.31*d[:, 1] < 0)
    E3 = np.stack([f.E1, f.E2, f.E3], 1); rows = []; cols = []
    for a, b in ((0, 1), (1, 2), (0, 2)): rows += [E3[:, a], E3[:, b]]; cols += [E3[:, b], E3[:, a]]
    A = sp.csr_matrix((np.ones(sum(len(r) for r in rows)), (np.concatenate(rows), np.concatenate(cols))), shape=(f.m, f.m)); A.data[:] = 1.0
    s.P = ((sp.identity(f.m) + A)/5.0).tocsr(); s.n = n; s.tc = 1.0; s.K = FK.build(f, L, s.P, n)
    r_ = np.roots([1, 0, -1, -h]); r_ = r_[np.isreal(r_)].real; psiP = r_.max()
    rng = np.random.default_rng(seed)
    z = np.sqrt(s0)*(1 + 0.01*(rng.standard_normal(f.m) + 1j*rng.standard_normal(f.m))); psi = psiP + 0.01*rng.standard_normal(f.n)
    s.S0 = float((np.abs(z)**2).sum()); s.m0 = s.S0/f.m
    return s, z, psi
def cls(f, L):
    d = f.xy[f.head] - f.xy[f.tail]; d = (d + L//2) % L - L//2; c = np.full(f.m, 2)
    c[d[:, 1] == 0] = 0; c[d[:, 0] == 0] = 1; return c
def run(L=48, s0=6.0, ac=1.0, tilt=1, Tt=100.0, dt=0.005, seed=1):
    s, z, psi = build(L, s0, ac, tilt=tilt, seed=seed); f = s.fab; c = cls(f, L); t = 0.0; i = 0; prev = None; log = []
    while t < Tt - 1e-9:
        z, psi = s.step(z, psi, dt); t += dt; i += 1
        if i % 200 == 0:
            rec, prev = TL.stats(f, z, prev); I = np.abs(z)**2
            rec.update(t=round(t, 1), fw=round(float((psi > 0).mean()), 4), Ic=[round(float(I[c == k].mean()), 3) for k in range(3)])
            log.append(rec)
    name = f"w64_L{L}_s{s0:g}_a{ac:g}_t{tilt}_s{seed}"; np.savez(name + ".npz", z=z, psi=psi); json.dump(log, open(name + ".json", "w"))
    for q in log[::5] + [log[-1]]: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in q.items()}, flush=True)
if __name__ == "__main__":
    kw = {k: (int(v) if k in ("L", "tilt", "seed") else float(v)) for k, v in (a.split("=") for a in sys.argv[1:])}; run(**kw)
