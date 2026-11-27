import os; os.environ["NB"]="0"
import numpy as np, json, flipfab as FF, simNlib as N, sim34d as S, local60 as LC, rnd51 as R
def build(q, L=96):
    f=FF.FlipFabric(L,q,0); s=LC.Loc.__new__(LC.Loc); s.fab=f; s.mod=S.M34(f,beta=1.5,eps0=3.0,h=0.3,kappa=1.0); s.wp=N.wplus(1.5); s.mod.wp=s.wp; s.lc=1.3; s.cs,s.cc=1+0j,1+1j
    d=np.load(f"rfl_L{L}_q{q:g}_g0_s1_n100_ell8.npz"); R.flip(f,d["bad"]); s.m0=3.5
    return f, s, d
out={}
for q in (0,0.3,3,10):
    f,s,d=build(q); c=s.F(d["z"],np.zeros(f.n),parts=True)["cost"]; M=d["psi"]<0
    dg=f.deg; rows=[]
    for k in range(3,15):
        sel=M&(dg==k)
        if sel.sum()>20: rows.append((k,int(sel.sum()),round(float(c[sel].mean()),1),round(float((c[sel]>-1).mean()),3)))
    out[q]=dict(cost=c[M].tolist(),deg=dg[M].tolist())
    print(q,'mean %.1f sd %.1f cheap %.3f'%(c[M].mean(),c[M].std(),(c[M]>-1).mean()),rows)
json.dump({str(k):v for k,v in out.items()},open("diagflip.json","w"))
