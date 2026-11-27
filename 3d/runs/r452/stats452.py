import sys, numpy as np
R="/mnt/project-files/model/s52"; sys.path[:0]=[R+"/r447",R+"/r396",R+"/r351",R]
import fastpatch; fastpatch.install()
import rq447 as W
a=0.75
def st(lat,seed):
    s,f,mod,QV,tqn,bad,tag = W.build(lat, seed, 4.0, 3.0, h=0.36, ac=0.25)
    S=np.array(mod.S); E=np.array(mod.E); sig=S.sum(0)            # sigma per loop
    k=mod.ke.astype(float)
    kq=np.array([k[E[j]] for j in range(4)])                       # k of each link in loop
    # self-pinning of loop q in mean field: sum_j 1/(a k_j)
    pin=(1.0/(a*kq)).sum(0)
    return sig,k,pin
for lat,lab in (("kun","Kun"),(R+"/r448/runs/R_e0.25_s1.npz","R1")):
  for seed in (1,2):
    sig,k,pin=st(lat,seed)
    vals,cnt=np.unique(np.abs(sig),return_counts=True)
    print(lab,seed,"|sigma| dist",dict(zip(vals.tolist(),np.round(cnt/cnt.sum(),3).tolist())),"k_e mean",k.mean().round(2),"k uniq",np.unique(k)[:8],"pin mean",pin.mean().round(3))
    for al in (0.1,0.2,0.25,0.3,0.4):
      lock=(al*np.abs(sig)<pin).mean(); print("   alpha",al,"locked frac (single-loop MF)",round(lock,3))
