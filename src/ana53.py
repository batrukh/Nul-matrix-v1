"""ana53: № 53 — розімкнені й замкнені шви (розд. 4.4 (129)). Для кожного прогону local60 sk=… :
кипіння і світи (t ≥ 150), де стоять світи (відстань вузлів світу до шва, до кінців, до зарядів-петель τ) проти рівномірного розподілу,
і закрученість ⟨sin h⟩ трикутників у колі r ≤ 6 біля кінців шва і біля зарядів (циклічних трикутників стріл)."""
import numpy as np, json, glob, sys
import local60 as M, rnd51 as R, seam51 as SM, birth48 as BR

L = 96; c = L//2; T0 = 150.0

def fabric(bad):
    s = M.Loc(L=L, h=0.3, lc=1.3); f = s.fab; R.flip(f, SM.seams(f)); R.flip(f, bad); return f

def cart(x, y): return x + 0.5*y, np.sqrt(3)/2*y

def tdist(px, py, X, Y):                        # найменша відстань на торі (декартові координати)
    best = None
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            d = np.hypot(X - px + a*L + b*L/2, Y - py + b*np.sqrt(3)/2*L); best = d if best is None else np.minimum(best, d)
    return best

def ends(sk, ln):                               # кінці шва в індексних координатах (x, y)
    a = ln//2
    if sk == 1 and ln < L: return [(c - a, c), (c + a, c)]
    if sk == 2: return [(c - a, c), (c + a, c), (c, c - a), (c, c + a)]
    return []

def one(fn):
    d = json.load(open(fn)); nm = d["name"]; sk = int(nm.split("_sk")[1].split("l")[0]); ln = int(nm.split("_sk")[1].split("l")[1].split("_")[0])
    seed = int(nm.split("_s")[2].split("_")[0]) if "_s" in nm else 0
    z0 = np.load(nm + ".npz"); bad = z0["bad"]; f = fabric(bad)
    nb = [[] for _ in range(f.n)]
    for p, q in zip(f.tail, f.head): nb[p].append(q); nb[q].append(p)
    S = np.stack([f.S1, f.S2, f.S3], 1); cyc = np.where(np.abs(S.sum(1)) == 3)[0]
    Xn, Yn = cart(f.xy[:, 0].astype(float), f.xy[:, 1].astype(float))
    TVx = Xn[f.TV].mean(1); TVy = Yn[f.TV].mean(1)            # (центри трикутників без розгортання; на краю тора неточні, тут не важливо)
    # заряди: центри циклічних трикутників і знак обходу
    ch = [(float(TVx[i]), float(TVy[i]), int(np.sign(S[i].sum()))) for i in cyc]
    # відстань вузла до шва (середина нитки шва)
    if bad.any():
        ex = (Xn[f.tail[bad]] + Xn[f.head[bad]])/2; ey = (Yn[f.tail[bad]] + Yn[f.head[bad]])/2
        dseam = np.min([tdist(x, y, Xn, Yn) for x, y in zip(ex, ey)], 0)
    else: dseam = None
    E = [cart(float(x), float(y)) for x, y in ends(sk, ln)]
    dend = np.min([tdist(x, y, Xn, Yn) for x, y in E], 0) if E else None
    dch = np.min([tdist(x, y, Xn, Yn) for x, y, _ in ch], 0) if ch else None
    sp = np.load(nm + "_sp.npz"); ts = sp["t"]; sel = ts >= T0 - 1e-6
    w2 = [r for r in d["w2"] if r["t"] >= T0]
    boil = float(np.mean([r["boil"] for r in w2])); nB = float(np.mean([r["nB"] for r in w2]))
    nw = []; big = []; dW = {"seam": [], "end": [], "ch": []}; hol = {"end": [], "ch": [], "all": [], "far": []}
    tri_d_end = np.min([tdist(x, y, TVx, TVy) for x, y in E], 0) if E else None
    tri_d_ch = np.min([tdist(x, y, TVx, TVy) for x, y, _ in ch], 0) if ch else None
    sgn_ch = None
    if ch:                                                    # знак найближчого заряду для кожного трикутника
        dd = np.array([tdist(x, y, TVx, TVy) for x, y, _ in ch]); sgn_ch = np.array([q[2] for q in ch])[np.argmin(dd, 0)]
    for k in np.where(sel)[0]:
        psi = sp["psi"][k]; z = sp["z"][k]; B = psi > 0
        cs = BR.comps(f, B, nb) if B.any() else []; nw.append(sum(1 for q in cs if q >= 20)); big.append(cs[0] if cs else 0)
        if B.any():
            if dseam is not None: dW["seam"].append(float(dseam[B].mean()))
            if dend is not None: dW["end"].append(float(dend[B].mean()))
            if dch is not None: dW["ch"].append(float(dch[B].mean()))
        h = np.sin(f.holonomy(z)); hol["all"].append(float(h.mean()))
        if tri_d_end is not None: hol["end"].append(float(h[tri_d_end <= 6].mean()))
        if tri_d_ch is not None: hol["ch"].append(float((h*sgn_ch)[tri_d_ch <= 6].mean()))   # закрученість біля заряду, помножена на його знак
        if dseam is not None:
            far = np.min(np.stack([dseam[f.TV[:, j]] for j in range(3)]), 0) > 15; hol["far"].append(float(h[far].mean()) if far.any() else np.nan)
    avg = lambda v: round(float(np.nanmean(v)), 3) if len(v) else None
    unif = dict(seam=round(float(dseam.mean()), 1) if dseam is not None else None, end=round(float(dend.mean()), 1) if dend is not None else None,
                ch=round(float(dch.mean()), 1) if dch is not None else None)
    fin = d["final"][:6]
    return dict(sk=sk, ln=ln, seed=seed, nbad=int(bad.sum()), ncharge=len(ch), charges=ch, boil=round(boil, 3), nB=round(nB), worlds=round(float(np.mean(nw)), 2),
                big=round(float(np.mean(big))), final=fin, dW={k: avg(v) for k, v in dW.items()}, dU=unif, hol={k: avg(v) for k, v in hol.items()})

if __name__ == "__main__":
    out = []
    for fn in sorted(glob.glob("lc_s03.5_n100_L96_s*_sk*.json")):
        try: r = one(fn)
        except Exception as e: print("skip", fn, e); continue
        out.append(r); print({k: v for k, v in r.items() if k != "charges"})
    json.dump(out, open("ana53.json", "w"), indent=1)
