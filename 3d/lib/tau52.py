"""tau52: стріли ниток із до-часу τ (черга № 41, розд. 4.4 (52)). Рішення автора 17.10.2026: нитка має стрілу (напрям процесу);
робоча гіпотеза — стріла вказує від вузла з меншим τ до вузла з більшим. Правильна трикутна ґратка L×L (тор), модель stall43
(β = 1,5, ε₀ = 3, λ_c = 1,3, α = 1). Поле τ:
  lin1 — τ = x; вертикальні нитки (вздовж y) рівності τ не мають і дивляться в +y (τ по них не міняється) → один шов (стовпець);
  lin2 — τ = x mod (L/2) → два шви;
  code — τ = x + 0,31·y → два поперечні шви (стрибки τ) по x і по y;
  rand — плавне випадкове періодичне τ з масштабом ℓ (гауссів фільтр білого шуму) → екстремуми й сідла τ.
Протокол — як regime51 при k = 1: ψ заморожене Tsmall + Ts (кипіння, m0 самоузгоджене), потім ψ відпущене на Tb."""
import numpy as np, sys, json, os
os.environ.setdefault("DELTA", "1e-6")
import stall43 as T, rnd51 as R, tile48 as TL, birth48 as BR

def tau_field(L, mode, ell=8.0, seed=0):
    x, y = np.divmod(np.arange(L*L), L)
    if mode == "lin1": return x.astype(float), "lin1"
    if mode == "lin2": return (x % (L//2)).astype(float), "lin2"
    if mode == "code": return x + 0.31*y, "code"
    rng = np.random.default_rng(700 + seed); w = rng.standard_normal((L, L))
    k = np.fft.fftfreq(L)*2*np.pi; K2 = k[:, None]**2 + k[None, :]**2
    t = np.real(np.fft.ifft2(np.fft.fft2(w)*np.exp(-K2*ell**2/2))); return t.ravel()/t.std(), f"rand{ell:g}"

def orient_tau(fab, tau, mode):
    L = fab.L; xa, ya = fab.xy[fab.tail].T; xb, yb = fab.xy[fab.head].T
    dx = (xb - xa + L//2) % L - L//2; dy = (yb - ya + L//2) % L - L//2
    dt = tau[fab.head] - tau[fab.tail]
    if mode in ("lin1", "lin2"): dt = np.where(dx == 0, dy.astype(float), dt)     # нитки вздовж y: τ не міняється, стріла +y
    R.flip(fab, dt < 0)
    nb = [[] for _ in range(fab.n)]
    for a, c in zip(fab.tail, fab.head): nb[a].append(c); nb[c].append(a)
    ext = sum(1 for v in range(fab.n) if all(tau[w] > tau[v] for w in nb[v]) or all(tau[w] < tau[v] for w in nb[v]))
    return ext, nb

def run(L=48, mode="rand", ell=8.0, gseed=1, seed=1, Tsmall=100.0, Ts=50.0, Tb=100.0, h=0.3, dt=0.005, lc=1.3):
    rng = np.random.default_rng(seed); r = np.roots([1, 0, -1, -h]); psiM = r[np.isreal(r)].real.min()
    s = T.Sys(L=L, h=h, lc=lc); f = s.fab
    tau, tag = tau_field(L, mode, ell, gseed); ext, nb = orient_tau(f, tau, mode)
    z = s.wp*(1 + 0.01*(rng.standard_normal(f.m) + 1j*rng.standard_normal(f.m))); psi = np.full(f.n, psiM)
    logS = []; z = TL.evolve(s, z, psi, Tsmall + Ts, dt, logS)
    s.m0 = float(np.mean([q["A2"] for q in logS if q["t"] > Tsmall + Ts/2]))
    psi = psiM + 0.01*rng.standard_normal(f.n); t = 0.0; i = 0; w2 = []; prev = None
    while t < Tb - 1e-9:
        z, psi = s.step(z, psi, dt); t += dt; i += 1
        if i % 200 == 0:
            B = psi > 0; cs = BR.comps(f, B, nb) if B.any() else []
            rec, prev = TL.stats(f, z, prev); rec.update(t=round(t, 1), nB=int(B.sum()), nuc=sum(1 for q in cs if q >= 5), big=cs[0] if cs else 0)
            w2.append(rec)
    B = psi > 0; final = BR.comps(f, B, nb) if B.any() else []
    name = f"tau_{tag}_L{L}_g{gseed}_s{seed}"
    np.savez(name + ".npz", z=z, psi=psi, tau=tau, m0=s.m0)
    return dict(name=name, L=L, mode=mode, ell=ell, ext=ext, m0=s.m0, boilS=logS[-1], logS=logS, w2=w2, final=final[:30])

if __name__ == "__main__":
    kw = {k: (v if k == "mode" else int(v) if k in ("L", "gseed", "seed") else float(v)) for k, v in (a.split("=") for a in sys.argv[1:])}
    d = run(**kw); json.dump(d, open(d["name"] + ".json", "w"))
    print(d["name"], "ext", d["ext"], "m0", round(d["m0"], 3), {k: round(v, 3) for k, v in d["boilS"].items()}, "final", d["final"][:12])
    for q in d["w2"][::10]: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in q.items()})
