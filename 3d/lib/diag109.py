import os; os.environ["NB"]="0"
import numpy as np, json, fab3d as F, simNlib as N, sim34d as S, local60 as LC, rnd51 as R
def build3(var):
    f=F.Fabric3D(16); fac=1.0 if var=="A" else 0.5
    s=LC.Loc.__new__(LC.Loc); s.fab=f; s.mod=S.M34(f,beta=1.5*fac,eps0=3.0*fac,h=0.3,kappa=fac); s.wp=N.wplus(3*fac); s.mod.wp=s.wp; s.lc=1.3*fac; s.cs,s.cc=1.0+0j,1+1j
    return s
out={}
for nm in ("r3d_L16_B_s1_n100_ell3_g0.02","r3d_L16_B_s1_n100_ell3_g0.02_v2","r3d_L16_B_s1_n100_ell3_g0.02_v4"):
    d=np.load(nm+".npz"); s=build3("B"); f=s.fab; R.flip(f,d["bad"]); s.m0=3.5
    c=s.F(d["z"],np.zeros(f.n),parts=True)["cost"]
    res=dict(mean=float(c.mean()),sd=float(c.std()),cheap=float((c>-1).mean()))
    if "_v" in nm:
        k=int(nm[-1]); cores=F.vortex_lines(16,k,1); ij=f.ijk[:,:2].astype(float); dist=np.full(f.n,99.)
        for a,b,q in cores:
            dd=(ij-np.array([a,b])+8)%16-8; dist=np.minimum(dist,np.hypot(*dd.T))
        res["near"]=dict(cheap=float((c[dist<1.5]>-1).mean()),mean=float(c[dist<1.5].mean()),n=int((dist<1.5).sum()))
        res["far"]=dict(cheap=float((c[dist>3]>-1).mean()),mean=float(c[dist>3].mean()))
    out[nm]=res; print(nm,res)
json.dump(out,open("diag109.json","w"))
