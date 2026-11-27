"""(452Б) скручення петель у світі і в Матриці на готових знімках (450: α0,1; 451: α0,25), ґратки R."""
import sys, glob, json, numpy as np
R="/mnt/project-files/model/s52"; sys.path[:0]=[R+"/r447",R+"/r396",R+"/r351",R]
import fastpatch; fastpatch.install()
import rq447 as W
out=[]
for run,ac in (("r450",0.1),("r451",0.25)):
  for k in range(1,9):
    f_=[x for x in glob.glob(f"{R}/{run}/runs/s{k}/*.npz") if not x.endswith("_ck.npz")]
    if not f_: continue
    d=np.load(f_[0]); z=d["z"]; psi=d["psi"]
    s,f,mod,QV,tqn,bad,tag=W.build(f"{R}/r448/runs/R_e0.25_s{k}.npz",k,4.0,3.0,h=0.36,ac=ac)
    _,P=mod.loop(z); tw=1-np.cos(np.angle(P)); inW=(psi[QV]>0).all(1); inM=(psi[QV]<0).all(1)
    r=dict(run=run,ac=ac,seed=k,twW=float(tw[inW].mean()) if inW.any() else None,twM=float(tw[inM].mean()) if inM.any() else None,nW=int(inW.sum()),nM=int(inM.sum()))
    out.append(r); print(r,flush=True)
json.dump(out,open("twist452b.json","w"),indent=1)
