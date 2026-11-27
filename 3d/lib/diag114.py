"""(114): розміри пласких ділянок (протосвітів) у кінцевому стані на 3D-сітках."""
import os; os.environ["NB"]="0"
import numpy as np, json, fabgen as G, rnd51 as R, birth48 as BR
out={}
for kind,sd in (("layer",1),("layer",2),("planar",1),("all",1)):
    f=G.FabricG(16,kind); d=np.load(f"rgen_{kind}_L16_s{sd}_n100.npz"); R.flip(f,d["bad"]); z=d["z"]
    hol=np.abs(f.holonomy(z)); hn=np.bincount(f.TV.ravel(),np.repeat(hol,3),f.n)/f.tnode
    A2n=(np.bincount(f.tail,np.abs(z)**2,f.n)+np.bincount(f.head,np.abs(z)**2,f.n))/f.deg
    nb=[[] for _ in range(f.n)]
    for p,q in zip(f.tail,f.head): nb[p].append(q); nb[q].append(p)
    M=(hn<0.35)&(d["psi"]<0); cs=BR.comps(f,M,nb) if M.any() else []
    out[f"{kind}_s{sd}"]=dict(flat=float(M.mean()),sizes=cs[:10],n_ge60=sum(c>=60 for c in cs),psi_flat=float(d["psi"][M].mean()) if M.any() else None,A2_flat=float(A2n[M].mean()) if M.any() else None,psi_all=float(d["psi"].mean()))
    print(kind,sd,out[f"{kind}_s{sd}"])
json.dump(out,open("diag114.json","w"))
