"""(111): чи є в 3D-Матриці «протосвіти» — пласкі ділянки ниток з інтенсивністю, близькою до вакууму світу (ознаки місць народження у 2D)."""
import os; os.environ["NB"]="0"
import numpy as np, json, local60 as LC, fab3d as F, rnd51 as R
def node_feats(f, z):
    n=f.n; a_,b_=f.tail,f.head; A2=np.abs(z)**2; cnt=np.bincount(a_,None,n)+np.bincount(b_,None,n)
    A=(np.bincount(a_,A2,n)+np.bincount(b_,A2,n))/cnt; hol=np.abs(f.holonomy(z)); tn=np.zeros(n); tc=np.zeros(n)
    for j in range(3): np.add.at(tn,f.TV[:,j],hol); np.add.at(tc,f.TV[:,j],1)
    return A, tn/tc
out={}
# 2D: Матриця (ψ < 0) у знімках t = 10..60, 4 реалізації
s=LC.Loc(L=96,h=0.3,lc=1.3); f=s.fab; t0,h0=f.tail.copy(),f.head.copy()
H2=[];A2_=[]
for sd in (1,2,3,4):
    d=np.load(os.environ.get("SNPAT","lc_s03.5_n100_L96_s{sd}_tau8_D1_g0.02_iso_sn_sn.npz").format(sd=sd))
    for k in range(10,60,5):
        tl=d["tail"][k]; fl=tl!=t0; f.tail=tl; f.head=np.where(fl,t0,h0)
        A,H=node_feats(f,d["z"][k].astype(complex)); M=d["psi"][k]<-0.3; H2+=list(H[M]); A2_+=list(A[M])
H2=np.array(H2); A2_=np.array(A2_)
out["2D"]=dict(hol_mean=float(H2.mean()), proto=float(((H2<0.35)&(A2_>5)).mean()), flat=float((H2<0.35).mean()), n=len(H2))
for nm,var in (("r3d_L16_B_s1_n100_ell3_g0.02","B"),("r3d_L16_B_s2_n100_ell3_g0.02","B"),("r3d_L16_A_s1_n100_ell3_g0.02","A")):
    if not os.path.exists(nm+".npz"): continue
    d=np.load(nm+".npz"); f3=F.Fabric3D(16); R.flip(f3,d["bad"]); A,H=node_feats(f3,d["z"])
    vac=6.85 if var=="B" else 34.0
    out[nm]=dict(hol_mean=float(H.mean()), proto=float(((H<0.35)&(A>5/6.85*vac)).mean()), flat=float((H<0.35).mean()), n=int(f3.n))
for k,v in out.items(): print(k,{q:round(x,4) if isinstance(x,float) else x for q,x in v.items()})
json.dump(out,open("diag111.json","w"))
