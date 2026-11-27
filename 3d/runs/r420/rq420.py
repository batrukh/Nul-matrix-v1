"""rq420 (17.11): латка Матриці в хаосі. rq383 (sgn=0, рівне α) на прискореному коді; фізика всередині латки без змін.
Латка — куля радіусом R (env R, шари) довкола центру тора; петля «в латці», якщо всі 4 її вершини в кулі.
COMP=1 (типово): нитки хаосу дістають амплітудну тягу пласкої Матриці, якої їм бракує без петель (M) або яку змінюють випадкові фази (F), щоб запас не перетікав у латку.
CHAOS=M: петлі поза латкою вимкнено (нитки й вузли є, геометрії нема — ні вирівнювання, ні ціни ε0).
CHAOS=F: петлі поза латкою мають випадкову заморожену фазу ph = e^{iθ}, θ ∈ [0, 2π) (seed 900+seed) — нитки не можуть скластися в пласкі петлі.
REG=1: латка і хаос зберігають запас окремо (обмін запасом через край вимкнено).
Профіль ψ і |z|² за радіусом щоодиниці — prof420.npz. Продовження з ck: лічильник кроків — з *_ck.npz."""
import os, sys, glob, numpy as np
R_ = "/mnt/project-files/model/s52"; sys.path[:0] = [R_ + "/r420", R_ + "/r396", R_ + "/r351", R_]
os.environ["NB"] = "0"
import fastpatch; fastpatch.install()
import qf2, qf3
g = {"__name__": "rq383x", "__file__": R_ + "/r383/rq383.py"}
exec(open(R_ + "/r383/rq383.py").read(), g)
LC = g["LC"]; Q = g["Q"]; _b = LC.Loc.step
RAD = float(os.environ["R"]); CH = os.environ["CHAOS"]; SEED = int(g["kw"]["seed"]); st = dict(i=0, T=[], P=[], Z=[], set=False)
PHA = None
COMP = int(os.environ.get("COMP", "1")); LW0 = None; REG = int(os.environ.get("REG", "0"))
def qf(z, E, S, QV, chi, dchi, eps0, d2, a, m, n):
    lw, br, fc = qf3.qforce3(z, E, S, QV, PHA, chi, dchi, eps0, d2, a, m, n)
    if LW0 is not None: lw = lw + LW0*z/np.sqrt(np.abs(z)**2 + d2)   # «тиск» без геометрії: лише амплітуда, фази не чіпає
    return lw, br, fc
def setup(s):
    global PHA
    f = s.fab; mod = s.mod; L = f.L; c = L/2 - 0.5
    d = (f.ijk - c + L/2) % L - L/2; r = np.linalg.norm(d, axis=1); st["r"] = r; st["rb"] = np.minimum(np.floor(r).astype(int), L)
    inq = (r[mod.QV] <= RAD).all(1); st["inq"] = inq
    E2 = np.ascontiguousarray(np.array(mod.E)); S2 = np.ascontiguousarray(np.array(mod.S)); QV = np.ascontiguousarray(mod.QV)
    th = np.random.default_rng(900 + SEED).uniform(0, 2*np.pi, len(QV)); PHF = np.where(inq, 1.0 + 0j, np.exp(1j*th))
    global LW0
    if COMP:   # компенсація: нитки хаосу отримують ту саму амплітудну тягу петель, що й у рівній пласкій Матриці (|z|² = s0), — без фаз і без ціни ε0
        s0 = float(g["kw"].get("s0", 3.0)); zu = np.full(f.m, np.sqrt(s0) + 0j); one = np.ones(f.n)
        lw_full = qf3.qforce3(zu, E2, S2, QV, np.ones(len(QV), complex), one, one, float(mod.eps0), mod.d2, mod.a, f.m, f.n)[0]
        if CH == "M": lw_in = qf3.qforce3(zu, np.ascontiguousarray(E2[:, inq]), np.ascontiguousarray(S2[:, inq]), np.ascontiguousarray(QV[inq]), np.ones(int(inq.sum()), complex), one, one, float(mod.eps0), mod.d2, mod.a, f.m, f.n)[0]
        else: lw_in = qf3.qforce3(zu, E2, S2, QV, PHF, one, one, float(mod.eps0), mod.d2, mod.a, f.m, f.n)[0]   # F: петлі хаосу з випадковою фазою дають іншу амплітудну тягу
        assert np.abs(lw_full.imag).max() < 1e-9
        LW0 = (lw_full - lw_in).real   # для F уявна частина lw_in (фазова фрустрація) лишається в динаміці   # частка тяги від петель, яких у латці немає
    if CH == "M":
        mod._E2 = np.ascontiguousarray(E2[:, inq]); mod._S2 = np.ascontiguousarray(S2[:, inq]); mod._QV = np.ascontiguousarray(QV[inq])
        PHA = np.ones(int(inq.sum()), np.complex128)
    else:
        mod._E2, mod._S2, mod._QV = E2, S2, QV
        PHA = PHF
    qf2.qforce2 = qf
    if REG:   # окреме збереження запасу: компенсація K·s не переносить запас через край латки (витік повертається рівномірно в межах своєї області)
        ta, hb = f.tail, f.head; lin = (r[ta] <= RAD) & (r[hb] <= RAD); st["lin"] = lin
        import fastpatch as FP; kr0 = FP._krfft
        def kreg(s_, src):
            out = np.empty_like(src)
            for msk in (lin, ~lin):
                k = kr0(s_, np.where(msk, src, 0.0)); out[msk] = k[msk] + (src[msk].sum() - k[msk].sum())/msk.sum()
            return out
        FP._krfft = kreg
        print(f"rq420: REG — окреме збереження, ниток у латці {lin.sum()} з {f.m}", flush=True)
    xm = (r[f.tail] + r[f.head])/2; st["lb"] = np.minimum(np.floor(xm).astype(int), L)
    print(f"rq420: CHAOS {CH} R {RAD} вузлів у латці {(r <= RAD).sum()} з {f.n}; петель у латці {inq.sum()} з {len(inq)}", flush=True)
def step(s, z, psi, dt):
    if not st["set"]:
        setup(s); st["set"] = True
        ck = glob.glob("*_ck.npz")
        if ck:
            st["i"] = int(np.load(ck[0], allow_pickle=True)["i"])
            if os.path.exists("prof420.npz"):
                P = np.load("prof420.npz"); k = P["t"] <= st["i"]*dt + 1e-9; st["T"], st["P"], st["Z"] = list(P["t"][k]), list(P["psi"][k]), list(P["z"][k])
            print("rq420: resume i", st["i"], flush=True)
    out = _b(s, z, psi, dt); st["i"] += 1
    if st["i"] % int(round(1/dt)) == 0:
        L = s.fab.L; A2 = np.abs(out[0])**2
        st["T"].append(st["i"]*dt); st["P"].append(np.bincount(st["rb"], out[1], L + 1)/np.maximum(np.bincount(st["rb"], None, L + 1), 1))
        st["Z"].append(np.bincount(st["lb"], A2, L + 1)/np.maximum(np.bincount(st["lb"], None, L + 1), 1))
        np.savez("prof420.npz", t=np.array(st["T"]), psi=np.array(st["P"]), z=np.array(st["Z"]))
    return out
LC.Loc.step = step
print(Q.run(**g["kw"]))
