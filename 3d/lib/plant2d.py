"""(110) контроль у 2D: та сама процедура посадки світу-диска в зрілу Матрицю (знімок L96), стріли заморожені. Рахуємо вузли ψ > 0 у межах 2R від центру."""
import os; os.environ["NB"] = "0"; os.environ.setdefault("DELTA", "1e-6")
import numpy as np, json, sys, scipy.sparse as sp
import local60 as LC, rnd51 as R, fftk as FK
from scipy.ndimage import distance_transform_edt as edt
def run(pr, Tt=20.0, pz=0, fp=0.0, dt=0.005, nm="lc_s03.5_n100_L96_s1_tau8_D1_g0.02_iso"):
    d = np.load(nm + ".npz"); L = 96; s = LC.Loc(L=L, h=0.3, lc=1.3); f = s.fab; R.flip(f, d["bad"])
    E3 = np.stack([f.E1, f.E2, f.E3], 1); rows = []; cols = []
    for a, b in ((0, 1), (1, 2), (0, 2)): rows += [E3[:, a], E3[:, b]]; cols += [E3[:, b], E3[:, a]]
    A = sp.csr_matrix((np.ones(sum(len(r) for r in rows)), (np.concatenate(rows), np.concatenate(cols))), shape=(f.m, f.m)); A.data[:] = 1.0
    s.P = ((sp.identity(f.m) + A)/5.0).tocsr(); s.n = 100; s.tc = 1.0; s.K = FK.build(f, L, s.P, 100)
    z = d["z"]; psi = d["psi"].copy(); s.S0 = float((np.abs(z)**2).sum()); s.m0 = float(d["m0"])
    W = (psi > 0).reshape(L, L); far = edt(~np.tile(W, (3, 3)))[L:2*L, L:2*L]; c = np.unravel_index(far.argmax(), far.shape)
    i, j = f.xy.T; di = (i - c[0] + L//2) % L - L//2; dj = (j - c[1] + L//2) % L - L//2
    rr = np.hypot(di + dj/2, dj*np.sqrt(3)/2); r_ = np.roots([1, 0, -1, -0.3]); psiW = r_[np.isreal(r_)].real.max()
    psi[rr <= pr] = psiW
    if pz: inn = (rr[f.tail] <= pr) & (rr[f.head] <= pr); z = z.copy(); z[inn] = (s.wp if pz == 2 else np.sqrt(np.mean(np.abs(z)**2)))
    zone = rr <= 2*pr + 3; out = []; t = 0.0; k = 0; FP = int(round(fp/dt)) if fp else 0; bal = []; import rungen as RG
    while t < Tt - 1e-9:
        if FP and k % FP == 0: bal.append(RG.fparts(s, f, z, psi, rr <= pr, t))   # (115): складники сили на ψ у посадженому диску
        z, psi = s.step(z, psi, dt); t += dt; k += 1
        if k % 200 == 0: out.append(dict(t=round(t, 1), n=int((psi[zone] > 0).sum()))); print(out[-1], flush=True)
    json.dump(dict(pr=pr, n0=int((rr <= pr).sum()), far=float(far.max()), w=out, **({"bal": bal} if FP else {})), open(("bal_" if FP else "") + f"plant2d_pr{pr:g}" + (f"_pz{pz}" if pz else "") + ".json", "w"))
if __name__ == "__main__": run(float(sys.argv[1]), pz=int(sys.argv[2]) if len(sys.argv) > 2 else 0, Tt=float(sys.argv[3]) if len(sys.argv) > 3 else 20.0, fp=float(sys.argv[4]) if len(sys.argv) > 4 else 0.0)
