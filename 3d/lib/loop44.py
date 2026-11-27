"""loop44: чи народження світу з хреста швів — артефакт тора (черга № 44, розд. 4.4 (55)). Правильна ґратка без швів коду, протокол — як clust53.
kind=loop:   замкнений шов-петля всередині Матриці: перевернуто горизонталі рядків y0, y0+a і вертикалі стовпців x0, x0+a квадрата a×a
             (петля стягується в точку, тор не обходить).
kind=almost: повний шов-рядок + шов-стовпець довжиною ln < L (майже хрест: стовпець не замкнений навколо тора).
kind=open:   відкрита Матриця (поглинальний пояс як open47: поза колом Rb = 0,433·L − W |z| → √m0, ψ → корінь Матриці);
             хрест двох відрізків довжиною 2·rs через центр, кінці — всередині поясу (для вільної зони шви «нескінченні», тор не обходять).
             ln = 0 — контроль без швів. m0 і світи — лише у вільній зоні r < Rb.
ac — поворот сили α (№ 45: ac = −1); gflip=1 — усі стріли Матриці навпаки. quad — вузли світів за чвертями хреста."""
import numpy as np, sys, json, os
os.environ.setdefault("DELTA", "1e-6")
import stall43 as T, rnd51 as R, tile48 as TL, birth48 as BR, seam51 as SM, simNlib as N

class Open(T.Sys):
    def __init__(s, W=12.0, gmax=1.0, **kw):
        super().__init__(**kw); f = s.fab
        s.r = N.disk(f, 0); s.Rb = 0.433*f.L - W
        s.gn = gmax*np.clip((s.r - s.Rb)/W, 0, 1)**2; s.ge = np.maximum(s.gn[f.tail], s.gn[f.head])
        rr = np.roots([1, 0, -1, -s.mod.h]); s.psiM = rr[np.isreal(rr)].real.min()
    def F(s, z, psi, parts=False):
        if parts: return super().F(z, psi, parts)
        dz, dpsi = super().F(z, psi); A = np.abs(z) + 1e-12
        return dz - s.ge*(A - np.sqrt(s.m0))*z/A, dpsi - s.gn*(psi - s.psiM)

def run(L=48, kind="loop", a=24, ln=46, rs=38, gseed=1, seed=1, Tsmall=100.0, Ts=50.0, Tb=100.0, h=0.3, dt=0.005, lc=1.3, W=12.0, ac=1.0, gflip=0):
    rng = np.random.default_rng(seed); r_ = np.roots([1, 0, -1, -h]); psiM = r_[np.isreal(r_)].real.min()
    s = Open(L=L, h=h, lc=lc, W=W, ac=ac) if kind == "open" else T.Sys(L=L, h=h, lc=lc, ac=ac); f = s.fab; R.flip(f, SM.seams(f))
    xa, ya = f.xy[f.tail].T; xb, yb = f.xy[f.head].T
    dy = (yb - ya + L//2) % L - L//2; dx = (xb - xa + L//2) % L - L//2; mx = np.minimum(xa, xb); my = np.minimum(ya, yb)
    H = (dy == 0) & (np.abs(dx) == 1); V = (dx == 0) & (np.abs(dy) == 1); c = L//2
    if kind == "loop":
        x0 = c - a//2; y0 = c - a//2
        bad = H & np.isin(ya, (y0, y0 + a)) & (mx >= x0) & (mx < x0 + a) | V & np.isin(xa, (x0, x0 + a)) & (my >= y0) & (my < y0 + a); tag = f"loop_{a}"
    elif kind == "almost":
        bad = H & (ya == c) | V & (xa == c) & (my >= c - ln//2) & (my < c - ln//2 + ln); tag = f"almost_{ln}"
    else:
        bad = (H & (ya == c) & (np.abs(mx + 0.5 - c) < rs) | V & (xa == c) & (np.abs(my + 0.5 - c) < rs)) if ln else np.zeros(f.m, bool)
        tag = f"open_{'x' if ln else '0'}{rs}"
    R.flip(f, bad)
    if gflip: R.flip(f, np.ones(f.m, bool))                       # (55)/№ 45: усі стріли навпаки (шви лишаються дефектами)
    if ac != 1.0 or gflip: tag += f"_a{ac:g}" + ("_gf" if gflip else "")
    free = s.r < s.Rb if kind == "open" else np.ones(f.n, bool); fe = free[f.tail] & free[f.head]
    nb = [[] for _ in range(f.n)]
    for p, q in zip(f.tail, f.head): nb[p].append(q); nb[q].append(p)
    z = s.wp*(1 + 0.01*(rng.standard_normal(f.m) + 1j*rng.standard_normal(f.m))); psi = np.full(f.n, psiM)
    logS = []; z = TL.evolve(s, z, psi, Tsmall + Ts/2, dt, logS); acc = []
    for k in range(10):
        l2 = []; z = TL.evolve(s, z, psi, Ts/20, dt, l2); logS += l2; acc.append(float((np.abs(z[fe])**2).mean()))
    s.m0 = float(np.mean(acc))
    psi = psiM + 0.01*rng.standard_normal(f.n); t = 0.0; i = 0; w2 = []; prev = None
    while t < Tb - 1e-9:
        z, psi = s.step(z, psi, dt); t += dt; i += 1
        if i % 200 == 0:
            B = (psi > 0) & free; cs = BR.comps(f, B, nb) if B.any() else []
            rec, prev = TL.stats(f, z, prev); rec.update(t=round(t, 1), nB=int(B.sum()), nuc=sum(1 for q_ in cs if q_ >= 5), big=cs[0] if cs else 0)
            w2.append(rec)
    B = (psi > 0) & free; final = BR.comps(f, B, nb) if B.any() else []
    rn = N.disk(f, 0); where = float(rn[B].mean()) if B.any() else None
    qx = np.sign(((f.xy[:, 0] - c + L//2) % L - L//2) + 0.5); qy = np.sign(((f.xy[:, 1] - c + L//2) % L - L//2) + 0.5)
    quad = {k: int((B & (qx == sx) & (qy == sy)).sum()) for k, sx, sy in (("++", 1, 1), ("+-", 1, -1), ("-+", -1, 1), ("--", -1, -1))}   # світи за чвертями (знак x−c, y−c)
    name = f"lp_{tag}_L{L}_g{gseed}_s{seed}"
    np.savez(name + ".npz", z=z, psi=psi, bad=bad, m0=s.m0)
    return dict(name=name, kind=kind, a=a, ln=ln, rs=rs, nbad=int(bad.sum()), nfree=int(free.sum()), m0=s.m0, boilS=logS[-1], logS=logS, w2=w2, final=final[:30], rmean=where, quad=quad, ac=ac, gflip=gflip)

if __name__ == "__main__":
    kw = {k: (v if k == "kind" else int(v) if k in ("L", "gseed", "seed", "ln", "a", "rs", "gflip") else float(v)) for k, v in (a.split("=") for a in sys.argv[1:])}
    d = run(**kw); json.dump(d, open(d["name"] + ".json", "w"))
    print(d["name"], "nbad", d["nbad"], "nfree", d["nfree"], "m0", round(d["m0"], 3), {k: round(v, 3) for k, v in d["boilS"].items()}, "final", d["final"][:12], "rmean", d["rmean"], "quad", d["quad"])
    for q in d["w2"][::10]: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in q.items()})
