"""diagp (118): складники сили на ψ у пласких ділянках Матриці (|гол.| < 0,35, ψ < 0) — 2D (знімок L96) проти 3D Б (кінцеві стани rq)."""
import os; os.environ["NB"] = "0"; os.environ.setdefault("DELTA", "1e-6")
import numpy as np, json, sys
import local60 as LC
def parts(s, f, z, psi, hn):
    M = (hn < 0.35) & (psi < 0); P = s.F(z, psi, parts=True)
    A2n = (np.bincount(f.tail, np.abs(z)**2, f.n) + np.bincount(f.head, np.abs(z)**2, f.n))/(np.bincount(f.tail, None, f.n) + np.bincount(f.head, None, f.n))
    out = dict(frac=float(M.mean()), psi=float(psi[M].mean()), A2=float(A2n[M].mean()), hol=float(hn[M].mean()))
    out.update({k: float(v[M].mean()) for k, v in P.items()}); out["sum"] = out["pot"] + out["grad"] + out["cost"] + out["coup"]
    out.update({"rest_" + k: float(v[(~M) & (psi < 0)].mean()) for k, v in P.items()}); out["rest_psi"] = float(psi[(~M) & (psi < 0)].mean()); return out
if sys.argv[1] == "2d":
    import rnd51 as R
    d = np.load("lc_s03.5_n100_L96_s1_tau8_D1_g0.02_iso.npz"); s = LC.Loc(L=96, h=0.3, lc=1.3); f = s.fab; R.flip(f, d["bad"]); s.m0 = float(d["m0"])
    z, psi = d["z"], d["psi"]; hol = np.abs(f.holonomy(z)); hn = np.bincount(f.TV.ravel(), np.repeat(hol, 3), f.n)/np.bincount(f.TV.ravel(), None, f.n)
else:
    import rungen4 as Q
    seed = int(sys.argv[2]); suf = sys.argv[3] if len(sys.argv) > 3 else ""; kw = dict(a.split("=") for a in sys.argv[4:]); s, f, mod, z, psi = Q.run(L=16, seed=seed, check=1, **{k: float(v) for k, v in kw.items()}); d = np.load(f"rq_L16_s{seed}_n100{suf}.npz"); z, psi = d["z"], d["psi"]
    s.m0 = float((np.abs(z)**2).mean()); hol = np.abs(f.holonomy(z)); hn = np.bincount(f.TV.ravel(), np.repeat(hol, 3), f.n)/f.tnode
print({k: round(v, 3) for k, v in parts(s, f, z, psi, hn).items()})
