import os; os.environ["NB"]="0"
import numpy as np, json, fab3d as F, simNlib as N, sim34d as S, local60 as LC, rnd51 as R
psiW=1.12541878; out={}
def summ(P,sel): return {k:round(float(v[sel].mean()),2) for k,v in P.items()} | {"total":round(float(sum(P.values())[sel].mean()),2)}
d=np.load("lc_s03.5_n100_L96_s1_tau8_D1_g0.02_iso.npz"); s=LC.Loc(L=96,h=0.3,lc=1.3); f=s.fab; R.flip(f,d["bad"]); s.m0=float(d["m0"])
P=s.F(d["z"],d["psi"],parts=True); W=d["psi"]>0.8; out["2D світ (справжній)"]=summ(P,W)
P=s.F(d["z"],np.full(f.n,psiW),parts=True); out["2D Матриця, ψ=ψсвіту"]=summ(P,d["psi"]<0)
for var in ("B","A"):
    fac=1.0 if var=="A" else 0.5; f3=F.Fabric3D(16); s3=LC.Loc.__new__(LC.Loc); s3.fab=f3
    s3.mod=S.M34(f3,beta=1.5*fac,eps0=3.0*fac,h=0.3,kappa=fac); s3.wp=N.wplus(2*1.5*fac); s3.mod.wp=s3.wp; s3.lc=1.3*fac; s3.cs,s3.cc=1+0j,1+1j
    d3=np.load(f"r3d_L16_{var}_s1_n100_ell3_g0.02.npz"); R.flip(f3,d3["bad"]); s3.m0=float((np.abs(d3["z"])**2).mean())
    P=s3.F(d3["z"],np.full(f3.n,psiW),parts=True); out[f"3D {var}, ψ=ψсвіту"]=summ(P,np.ones(f3.n,bool))
for k,v in out.items(): print(k,v)
json.dump(out,open("diag110.json","w"),ensure_ascii=False)
