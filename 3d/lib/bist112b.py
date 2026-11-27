"""(112б): медіанний дрейф закрученості і ймовірність лишитися пласким через 1, 5, 10 од. (вузли Матриці далі 4 від світів)."""
import os; os.environ["NB"]="0"
import numpy as np, json, local60 as LC, contextlib, io
from scipy.ndimage import distance_transform_edt as edt
with contextlib.redirect_stdout(io.StringIO()): from diag111 import node_feats
L=96; s=LC.Loc(L=L,h=0.3,lc=1.3); f=s.fab; t0,h0=f.tail.copy(),f.head.copy()
Hb=np.linspace(0,np.pi,41); cen=(Hb[:-1]+Hb[1:])/2; acc={k:[[] for _ in range(40)] for k in (1,5,10)}
stay={k:[0,0] for k in (1,5,10)}; base=[0,0]; enter={k:[0,0] for k in (1,5,10)}
for sd in (1,2,3,4):
    d=np.load(f"lc_s03.5_n100_L96_s{sd}_tau8_D1_g0.02_iso_sn_sn.npz"); PS=d["psi"]; T=len(d["t"]); HH=[]
    for k in range(T):
        tl=d["tail"][k]; fl=tl!=t0; f.tail=tl; f.head=np.where(fl,t0,h0); HH.append(node_feats(f,d["z"][k].astype(complex))[1])
    HH=np.array(HH); far=np.array([edt(~np.tile((PS[k]>0).reshape(L,L),(3,3)))[L:2*L,L:2*L].reshape(-1)>4 for k in range(T)]); M=(PS<-0.3)&far
    for k in range(10,T):
        base[0]+=int((HH[k][M[k]]<0.35).sum()); base[1]+=int(M[k].sum())
        for lag in (1,5,10):
            if k+lag>=T: continue
            m=M[k]&M[k+lag]; b=np.clip(np.digitize(HH[k][m],Hb)-1,0,39); dh=HH[k+lag][m]-HH[k][m]
            for i in np.unique(b): acc[lag][i]+=list(dh[b==i][::7])
            fl_=m&(HH[k]<0.35); stay[lag][0]+=int((HH[k+lag][fl_]<0.35).sum()); stay[lag][1]+=int(fl_.sum())
            bo=m&(HH[k]>1.0); enter[lag][0]+=int((HH[k+lag][bo]<0.35).sum()); enter[lag][1]+=int(bo.sum())
res=dict(h=cen.tolist(), med={lag:[float(np.median(a)) if len(a)>30 else None for a in acc[lag]] for lag in acc},
         stay={lag:v[0]/v[1] for lag,v in stay.items()}, enter={lag:v[0]/v[1] for lag,v in enter.items()}, base=base[0]/base[1])
json.dump(res,open("bist112b.json","w"))
print("base flat fraction",round(res["base"],3)); print("P(stay flat)",{k:round(v,3) for k,v in res["stay"].items()}); print("P(boil->flat)",{k:round(v,4) for k,v in res["enter"].items()})
for lag in (1,5): print("median drift lag",lag,[(round(h,2),round(x,3)) for h,x in zip(cen,res["med"][lag]) if x is not None][::2])
