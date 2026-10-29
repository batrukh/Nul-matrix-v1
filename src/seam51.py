"""seam51: чи кипіння Матриці на правильній ґратці засіяне «швами» напрямків ниток (розд. 4.4 (51)).
У simNlib.Fabric напрям нитки задається порядком створення: майже всі нитки дивляться в один бік, крім рядка 0 (горизонталі)
і стовпця 0 (вертикалі), де напрям протилежний. З поворотом сили (α ≠ 0) напрям фізичний, тож шви — дефекти.
mode=fresh: правильна ґратка без швів (шви перевернуто), старт із майже спокійної Матриці, ψ заморожене, T одиниць.
mode=keep:  те саме зі швами (контроль).
mode=heal:  спершу T/2 зі швами (Матриця закипає), потім шви перевертаються (z → z̄ на них, той самий фізичний стан) і ще T/2."""
import numpy as np, sys, json, os
os.environ.setdefault("DELTA", "1e-6")
import stall43 as T, rnd51 as R, tile48 as TL

def seams(fab):
    L = fab.L; xa, ya = fab.xy[fab.tail].T; xb, yb = fab.xy[fab.head].T
    dx = (xb - xa + L//2) % L - L//2; dy = (yb - ya + L//2) % L - L//2
    return (dx == 1) & (dy == 0) | (dx == 0) & (dy == -1)        # проти напрямку решти ниток свого сімейства

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
