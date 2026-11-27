import os; os.environ["NB"]="0"
import numpy as np, sys, run3d, fab3d as F, simNlib as N, sim34d as S, local60 as LC, rnd51 as R, stall43 as T
def build3(L, var, seed, eps_fac=None):
    f=F.Fabric3D(L); fac=1.0 if var=="A" else 0.5; ef=eps_fac if eps_fac else fac
    s=LC.Loc.__new__(LC.Loc); s.fab=f; s.mod=S.M34(f,beta=1.5*fac,eps0=3.0*ef,h=0.3,kappa=fac); s.wp=N.wplus(3*fac); s.mod.wp=s.wp; s.lc=1.3*fac; s.cs,s.cc=1.0+0j,1+1j
    return s
for var in ("B",):
    d=np.load(f"r3d_L16_{var}_s1_n100_ell3_g0.02.npz"); s=build3(16,var,1); f=s.fab; R.flip(f,d["bad"]); z=d["z"]; s.m0=float((np.abs(z)**2).mean())
    for psi_val in (-1.0, 0.0, 1.0):
        psi=np.full(f.n,psi_val); P=s.F(z,psi,parts=True)
        print(var,"psi",psi_val,{k:round(float(np.mean(v)),3) for k,v in P.items()})
# 2D reference
s2=LC.Loc(L=48,h=0.3,lc=1.3); f2=s2.fab
rng=np.random.default_rng(0)
import glob
fn=sorted(glob.glob("lc_s03.5_n100_L96_s1_tau8_D1_g0.02_iso.npz"))
print(fn)
