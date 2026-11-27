"""rq447 (20.11, (447)): модель rq351f/rq383g (sgn=0, рівномірне α) на ДОВІЛЬНІЙ ґратці з fabpts.
lat=kun        — тетраедри Куна L³ (контроль модуля);
lat=<file.npz> — точки P (n×3) і box (вирощені з поля PFC (421)); сусіди = грані Вороного, петлі = 3 четвірки тетраедра Делоне.
Що інакше, ніж у rq351f, і чому:
 • нитки/петлі/трикутники — з тетраедрів (на Куні — ті самі множини);
 • нормування на вузол — середнє (tq = петель на вузол, dg = ниток на вузол), бо на неправильній ґратці вони різні;
   β на нитку — як і раніше 2β/(a·k_e) для КОЖНОЇ нитки (k_e — її петлі);
 • місцевий закон K = P^n — n множень на розріджену P (FFT лише на правильній ґратці); P = I − (D − A)/(1 + d_max) — як fabgen.local_P;
 • τ — те саме поле tau_cubic, трилінійно інтерпольоване в точки (на Куні — точно).
Решта (сила петель qf2, _pre з fastpatch, RK4, dt, діагностика, ck/resume, EARLY) — без змін."""
import os; os.environ["NB"] = "0"; os.environ.setdefault("DELTA", "1e-6")
import sys, json, numpy as np, numba as nb
R = "/mnt/project-files/model/s52"; sys.path[:0] = [R + "/r447", R + "/r396", R + "/r351", R]
import fastpatch; fastpatch.install()
import rq351f as Q, fabgen as G, simNlib as N, local60 as LC, rnd51 as RN, tile48 as TL, birth48 as BR, fabpts as FP

@nb.njit(cache=True)
def _powmv(indptr, indices, data, x, n):
    y = x.copy(); t = np.empty_like(x)
    for _ in range(n):
        for i in range(len(indptr) - 1):
            acc = 0.0
            for p in range(indptr[i], indptr[i + 1]): acc += data[p]*y[indices[p]]
            t[i] = acc
        y, t = t, y
    return y

def build(lat, seed, ell, s0, h=0.3, lc=1.3, beta=1.3, eps0=3.0, kappa=1.0, ac=0.1, n=100):
    if lat == "kun": f = FP.fromKun(16); tag = "kun"
    else: d = np.load(lat); f = FP.fromPoints(d["P"], float(d["box"])); tag = os.path.basename(lat)[:-4]
    wp = N.wplus(beta); s = LC.Loc.__new__(LC.Loc); s.fab = f
    tau = FP.tau_points(f, ell, seed); bad = tau[f.head] - tau[f.tail] < 0; RN.flip(f, bad)
    QV = f.quads(); tqn = np.bincount(QV.ravel(), None, f.n); tq = tqn.mean(); dg = f.deg.mean()
    mod = Q.MQ(f, QV, 0.0, eps0*8/tq, h, kappa*6/dg, wp); mod.links(); mod.beta = 2*beta/(mod.a*mod.ke)
    s.mod = mod; s.wp = wp; s.lc = lc*6/dg; s.cs, s.cc = 1.0 + 0j, 1 + 1j*ac
    P = G.local_P(f).tocsr(); s.P = P; s.n = n; s.tc = 1.0
    ip, ix, dt_ = P.indptr.astype(np.int64), P.indices.astype(np.int64), P.data.astype(float)
    s.K = lambda src: _powmv(ip, ix, dt_, np.ascontiguousarray(src, float), n)
    return s, f, mod, QV, tqn, bad, tag

def run(lat="kun", seed=1, Tt=300.0, dt=0.005, ell=4.0, s0=3.0, h=0.3, ac=0.1, ck=5.0, early=0.0, check=0):
    s, f, mod, QV, tqn, bad, tag = build(lat, seed, ell, s0, h=h, ac=ac)
    nb_ = [[] for _ in range(f.n)]
    for p, q in zip(f.tail, f.head): nb_[p].append(q); nb_[q].append(p)
    rng = np.random.default_rng(seed); r_ = np.roots([1, 0, -1, -h]); psiM = r_[np.isreal(r_)].real.min()
    z = np.sqrt(s0)*(1 + 0.01*(rng.standard_normal(f.m) + 1j*rng.standard_normal(f.m))); psi = psiM + 0.01*rng.standard_normal(f.n)
    if check: return s, f, mod, z, psi
    s.S0 = float((np.abs(z)**2).sum()); s.m0 = s.S0/f.m
    name = f"w447_{tag}_s{seed}"; t = 0.0; i = 0; ST = int(round(1/dt)); w2 = []; prev = None; CK = int(round(ck/dt)); ckf = name + "_ck.npz"; hist = []
    if os.path.exists(ckf):
        c = np.load(ckf, allow_pickle=True); z = c["z"]; psi = c["psi"]; t = float(c["t"]); i = int(c["i"]); s.S0 = float(c["S0"]); s.m0 = s.S0/f.m
        w2 = list(c["w2"]); prev = c["prev"] if c["prev"].size else None; print("resume at", t, flush=True)
    print("n", f.n, "m", f.m, "deg", np.unique(f.deg, return_counts=True), "loops/node mean", tqn.mean(), "loops/link", np.unique(mod.ke), flush=True)
    while t < Tt - 1e-9:
        z, psi = s.step(z, psi, dt); t += dt; i += 1
        if not np.isfinite(psi).all() or not np.isfinite(z).all(): w2.append(dict(t=round(t, 2), blowup=True)); break
        if i % ST == 0:
            rec, prev = TL.stats(f, z, prev); B = psi > 0; cs = BR.comps(f, B, nb_) if B.any() else []
            hol = np.abs(f.holonomy(z)); hn = np.bincount(f.TV.ravel(), np.repeat(hol, 3), f.n)/f.tnode
            A2n = (np.bincount(f.tail, np.abs(z)**2, f.n) + np.bincount(f.head, np.abs(z)**2, f.n))/f.deg
            _, Pq = mod.loop(z); qh = np.abs(np.angle(Pq)); qn = np.bincount(QV.ravel(), np.repeat(qh, 4), f.n)/tqn
            rec.update(t=round(t, 1), nB=int(B.sum()), nuc=sum(1 for q in cs if q >= 5), big=cs[0] if cs else 0, top=cs[:6],
                       flat=float((hn < 0.35).mean()), proto=float(((hn < 0.35) & (A2n > 5)).mean()), holm=float(hn.mean()),
                       qhol=float(qh.mean()), qflat=float((qn < 0.35).mean()))
            w2.append(rec); print(rec, flush=True)
            if early:
                hist.append((rec["t"], rec["big"])); old = [b for tt, b in hist if tt <= rec["t"] - 50]
                if rec["big"] > early*f.n and old and abs(rec["big"] - old[-1]) < 0.01*f.n:
                    print("EARLY_STOP t", rec["t"], flush=True); open("early", "w").write(str(rec["t"])); break
        if CK and i % CK == 0:
            np.savez(ckf + ".tmp.npz", z=z, psi=psi, t=t, i=i, S0=s.S0, w2=np.array(w2, dtype=object), prev=prev if prev is not None else np.zeros(0)); os.replace(ckf + ".tmp.npz", ckf)
    np.savez(name + ".npz", z=z, psi=psi, bad=bad)
    json.dump(dict(name=name, lat=lat, seed=seed, s0=s0, ell=ell, ac=ac, h=h, n=f.n, m=f.m, nloop=len(QV), w2=w2), open(name + ".json", "w")); return name

if __name__ == "__main__":
    kw = {k: (v if k == "lat" else (int(v) if k in ("seed", "check") else float(v))) for k, v in (a.split("=") for a in sys.argv[1:])}
    print(run(**kw))
