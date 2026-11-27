"""sparse52: рідкісні «дефектні» нитки (ідея автора 17.10.2026, розд. 4.4 (53)). Правильна ґратка L×L без швів: усі стріли
кожного з трьох сімейств ниток дивляться в один бік; частка p ниток (випадково) перевернута. Модель stall43 без змін.
Протокол — як tau52: ψ заморожене Tsmall + Ts (кипіння, m0 самоузгоджене), потім ψ відпущене на Tb."""
import numpy as np, sys, json, os
os.environ.setdefault("DELTA", "1e-6")
import stall43 as T, rnd51 as R, tile48 as TL, birth48 as BR, seam51 as SM

def run(L=48, p=0.01, gseed=1, seed=1, Tsmall=100.0, Ts=50.0, Tb=100.0, h=0.3, dt=0.005, lc=1.3):
    rng = np.random.default_rng(seed); r = np.roots([1, 0, -1, -h]); psiM = r[np.isreal(r)].real.min()
    s = T.Sys(L=L, h=h, lc=lc); f = s.fab; R.flip(f, SM.seams(f))                  # прибрати шви коду
    bad = np.random.default_rng(900 + gseed).random(f.m) < p; R.flip(f, bad)         # рідкісні перевернуті нитки
    nb = [[] for _ in range(f.n)]
    for a, c in zip(f.tail, f.head): nb[a].append(c); nb[c].append(a)
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
    name = f"sparse_p{p:g}_L{L}_g{gseed}_s{seed}"
    np.savez(name + ".npz", z=z, psi=psi, bad=bad, m0=s.m0)
    return dict(name=name, L=L, p=p, nbad=int(bad.sum()), m0=s.m0, boilS=logS[-1], logS=logS, w2=w2, final=final[:30])

if __name__ == "__main__":
    kw = {k: (int(v) if k in ("L", "gseed", "seed") else float(v)) for k, v in (a.split("=") for a in sys.argv[1:])}
    d = run(**kw); json.dump(d, open(d["name"] + ".json", "w"))
    print(d["name"], "nbad", d["nbad"], "m0", round(d["m0"], 3), {k: round(v, 3) for k, v in d["boilS"].items()}, "final", d["final"][:12])
    for q in d["w2"][::10]: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in q.items()})
