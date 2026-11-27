import os,sys,glob,numpy as np
from collections import deque
os.environ["NB"]="0"; R="/mnt/project-files/model/s52"; sys.path[:0]=[R+"/r396",R+"/r351",R]
import fastpatch; fastpatch.install(); import rq351f as Q
s,f,mod,_,_=Q.run(L=16,seed=1,s0=3.0,h=0.36,ac=0.1,check=1); ta,hd=f.tail,f.head; n=f.n
nb=[[] for _ in range(n)]
for p,q in zip(ta,hd): nb[p].append(q); nb[q].append(p)
for t in (3,6,10,15,20,25):
    c=np.load(f"runs/r5_h0.36_e0.185_s1_am_snap{t}.npz"); z,psi,act=c["z"],c["psi"],c["act"]; ae=act[ta]&act[hd]; A2=np.abs(z)**2
    num=np.bincount(ta,A2*ae,n)+np.bincount(hd,A2*ae,n); den=np.bincount(ta,ae,n)+np.bincount(hd,ae,n); rn=num/np.maximum(den,1)
    d=np.full(n,99); dq=deque(np.flatnonzero(~act)); d[~act]=0
    while dq:
        u=dq.popleft()
        for w in nb[u]:
            if d[w]>d[u]+1: d[w]=d[u]+1; dq.append(w)
    M=act&(psi<0)
    print(f"t{t} nlat {act.sum()} світ {(act&(psi>0)).sum()}", " | ".join(f"d{k}: ρвузла {rn[M&(d==k)].mean():.2f} (живих за 2,66: {(rn[M&(d==k)]>2.66).mean()*100:.0f}%)" for k in (1,2,3,4) if (M&(d==k)).any()))
