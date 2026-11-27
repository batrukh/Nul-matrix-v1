"""(110): швидкість, з якою пласкі нитки (z = w₊, голономія 0) втрачають пласкість під дією повороту α, у 2D і 3D (ψ = ψ світу всюди)."""
import os; os.environ["NB"]="0"
import numpy as np, scipy.sparse as sp, fab3d as F, simNlib as N, sim34d as S, local60 as LC, rnd51 as R
psiW=1.12541878
def rate(s, f, ac):
    s.cc = 1 + 1j*ac; z = np.full(f.m, s.wp + 0j); psi = np.full(f.n, psiW); s.S0 = float((np.abs(z)**2).sum()); s.m0 = s.S0/f.m
    dz, _ = s.F(z, psi); ph = np.imag(dz/z)                                # dφ_e/dt
    hr = ph[f.E1]*f.S1 + ph[f.E2]*f.S2 + ph[f.E3]*f.S3; return float(np.abs(hr).mean()), float(np.abs(ph).mean())
def P_of(f, deg):
    E3 = np.stack([f.E1, f.E2, f.E3], 1); rows = []; cols = []
    for a, b in ((0, 1), (1, 2), (0, 2)): rows += [E3[:, a], E3[:, b]]; cols += [E3[:, b], E3[:, a]]
    A = sp.csr_matrix((np.ones(sum(len(r) for r in rows)), (np.concatenate(rows), np.concatenate(cols))), shape=(f.m, f.m)); A.data[:] = 1.0
    return ((sp.identity(f.m) + A)/(1 + deg)).tocsr()
d = np.load("lc_s03.5_n100_L96_s1_tau8_D1_g0.02_iso.npz"); s = LC.Loc(L=96, h=0.3, lc=1.3); f = s.fab; R.flip(f, d["bad"]); s.P = P_of(f, 4); s.n = 100; s.tc = 1.0; s.K = None
for ac in (0, 1): print("2D ac", ac, "|dhol/dt|, |dφ/dt| =", rate(s, f, ac))
d3 = np.load("r3d_L16_B_s1_n100_ell3_g0.02.npz"); f3 = F.Fabric3D(16); s3 = LC.Loc.__new__(LC.Loc); s3.fab = f3
s3.mod = S.M34(f3, beta=0.75, eps0=1.5, h=0.3, kappa=0.5); s3.wp = N.wplus(1.5); s3.mod.wp = s3.wp; s3.lc = 0.65; s3.cs = 1 + 0j
R.flip(f3, d3["bad"]); s3.P = F.local_P(f3); s3.n = 100; s3.tc = 1.0; s3.K = F.fft_K(f3, s3.P, 100)
for ac in (0, 1): print("3D B ac", ac, "|dhol/dt|, |dφ/dt| =", rate(s3, f3, ac))
