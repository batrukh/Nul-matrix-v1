"""clust53: згущення дефектних ниток в упорядкованій Матриці (черга № 43, розд. 4.4 (54)). Правильна ґратка L×L без швів.
kind=patch: у колі радіуса r (від центру) перевернуто частку q ниток (q = 1 — уся пляма, межа плями — стінка стріл; q = 0,5 — хаос);
kind=line:  відрізок довжиною len горизонтальних ниток одного рядка перевернуто (як шов коду, але скінченний; len = L — повний шов).
kind=cross: line + такий самий відрізок вертикальних ниток стовпця L/2 (два сімейства); kind=rows: len повних швів горизонталей.
Модель stall43 без змін, протокол — як sparse52."""
import numpy as np, sys, json, os
os.environ.setdefault("DELTA", "1e-6")
import stall43 as T, rnd51 as R, tile48 as TL, birth48 as BR, seam51 as SM, simNlib as N

def run(L=48, kind="patch", r=4.0, q=1.0, ln=12, gseed=1, seed=1, Tsmall=100.0, Ts=50.0, Tb=100.0, h=0.3, dt=0.005, lc=1.3):
    rng = np.random.default_rng(seed); r_ = np.roots([1, 0, -1, -h]); psiM = r_[np.isreal(r_)].real.min()
    s = T.Sys(L=L, h=h, lc=lc); f = s.fab; R.flip(f, SM.seams(f))
    if kind == "patch":
        rn = N.disk(f, 0); re = (rn[f.tail] + rn[f.head])/2
        bad = (re < r) & (np.random.default_rng(900 + gseed).random(f.m) < q); tag = f"patch_r{r:g}_q{q:g}"
    else:
        xa, ya = f.xy[f.tail].T; xb, yb = f.xy[f.head].T; dy = (yb - ya + L//2) % L - L//2; dx = (xb - xa + L//2) % L - L//2
        x0 = (L - ln)//2; bad = (dy == 0) & (ya == L//2) & (np.minimum(xa, xb) >= x0) & (np.minimum(xa, xb) < x0 + ln); tag = f"line_{ln}"
        if kind == "cross":                                        # + відрізок вертикальних ниток стовпця L/2 (друге сімейство)
            bad |= (dx == 0) & (xa == L//2) & (np.minimum(ya, yb) >= x0) & (np.minimum(ya, yb) < x0 + ln); tag = f"cross_{ln}"
        if kind == "rows":                                         # ln повних швів горизонталей, рівномірно
            bad = (dy == 0) & np.isin(ya, (np.arange(ln)*L)//ln + L//2 % (L//max(ln, 1))); tag = f"rows_{ln}"
    R.flip(f, bad)
    nb = [[] for _ in range(f.n)]
    for a, c in zip(f.tail, f.head): nb[a].append(c); nb[c].append(a)
    z = s.wp*(1 + 0.01*(rng.standard_normal(f.m) + 1j*rng.standard_normal(f.m))); psi = np.full(f.n, psiM)
    logS = []; z = TL.evolve(s, z, psi, Tsmall + Ts, dt, logS)
    s.m0 = float(np.mean([q_["A2"] for q_ in logS if q_["t"] > Tsmall + Ts/2]))
    psi = psiM + 0.01*rng.standard_normal(f.n); t = 0.0; i = 0; w2 = []; prev = None
    while t < Tb - 1e-9:
        z, psi = s.step(z, psi, dt); t += dt; i += 1
        if i % 200 == 0:
            B = psi > 0; cs = BR.comps(f, B, nb) if B.any() else []
            rec, prev = TL.stats(f, z, prev); rec.update(t=round(t, 1), nB=int(B.sum()), nuc=sum(1 for q_ in cs if q_ >= 5), big=cs[0] if cs else 0)
            w2.append(rec)
    B = psi > 0; final = BR.comps(f, B, nb) if B.any() else []
    rn = N.disk(f, 0); where = float(rn[B].mean()) if B.any() else None
    name = f"cl_{tag}_L{L}_g{gseed}_s{seed}"
    np.savez(name + ".npz", z=z, psi=psi, bad=bad, m0=s.m0)
    return dict(name=name, kind=kind, r=r, q=q, ln=ln, nbad=int(bad.sum()), m0=s.m0, boilS=logS[-1], logS=logS, w2=w2, final=final[:30], rmean=where)

if __name__ == "__main__":
    kw = {k: (v if k == "kind" else int(v) if k in ("L", "gseed", "seed", "ln") else float(v)) for k, v in (a.split("=") for a in sys.argv[1:])}
    d = run(**kw); json.dump(d, open(d["name"] + ".json", "w"))
    print(d["name"], "nbad", d["nbad"], "m0", round(d["m0"], 3), {k: round(v, 3) for k, v in d["boilS"].items()}, "final", d["final"][:12], "rmean", d["rmean"])
    for q in d["w2"][::10]: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in q.items()})
