"""Ідеї І-1(а) і І-4 на готових кінцевих станах t300: запас натягу Σ|z|²/S0, |z|² Матриці (нитки з ψ<0 на обох кінцях)
далеко від меж (відстань ≥3 шари) і біля меж, у доменах зі світом і без; для островів — розмір проти |z|² Матриці свого домену."""
import sys, glob, json, numpy as np; sys.argv=['x']; sys.path[:0]=['/mnt/project-files/model/s52/r351','/mnt/project-files/model/s52']
import rq351f as Q
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
s,f,mod,z0,p0=Q.run(L=16,check=1); N=4096; T,H=f.tail,f.head; L=16
xt=f.ijk[T,0].astype(float); d=(f.ijk[H,0]-xt+L/2)%L-L/2; xm=(xt+d/2)%L      # середина нитки по x
S0={}
def s0(sd):
    if sd not in S0:
        _,_,_,z,_=Q.run(L=16,seed=sd,check=1,s0=3.0,ell=4.0); S0[sd]=float((np.abs(z)**2).sum())
    return S0[sd]
R='/mnt/project-files/model/s52/'
runs=[(6,R+f'r391/runs/sp6_seed{k}',k) for k in range(1,17)]+[(4,R+f'r391/runs/sp4_seed{k}',k) for k in range(1,9)]\
    +[(8,R+f'r385/runs/a0.2_seed{k}',k) for k in range(1,9)]+[(12,R+f'r387/runs/seed{k}',k) for k in range(1,9)]\
    +[(16,R+f'r375/a0.2_seed{k}_T300',k) for k in range(9,17)]+[(16,R+f'r374/a0.2_seed{k}',k) for k in range(1,9)]
out=[]
for sp,dd,sd in runs:
    fn=glob.glob(dd+'/t3_*_ell4.npz')
    if not fn: print('немає',dd); continue
    c=np.load(fn[0]); z=c['z']; p=c['psi']; A=np.abs(z)**2; nw=p>0; wl=nw[T]&nw[H]
    g=coo_matrix((np.ones(wl.sum()),(T[wl],H[wl])),shape=(N,N)); nc,lab=connected_components(g,directed=False)
    sz=np.bincount(lab[nw],minlength=nc); big=int(sz.max()) if nw.any() else 0
    mat=(p[T]<0)&(p[H]<0)
    r=dict(split=sp,run=dd.split('/')[-1],seed=sd,big=big,budget=float(A.sum()/s0(sd)),Amat=float(A[mat].mean()))
    if sp<16:
        # відстань середини нитки до найближчої межі (межі x≈sp−½ і 15,5)
        dw=np.minimum(np.abs(((xm-(sp-.5))+8)%16-8),np.abs(((xm-15.5)+8)%16-8)); inA=xm<sp-.5
        for nm,dm,wd in (('A',inA,sp),('B',~inA,16-sp)):
            far=mat&dm&(dw>=3); near=mat&dm&(dw<1.5)
            Wd=[cc for cc in np.where(sz>=50)[0] if (((nw&(lab==cc))&(f.ijk[:,0]<sp if nm=='A' else f.ijk[:,0]>=sp)).sum()>=.8*sz[cc])]
            r[nm]=dict(width=wd,world=int(sum(sz[cc] for cc in Wd)),Afar=float(A[far].mean()) if far.any() else None,Anear=float(A[near].mean()) if near.any() else None)
    out.append(r); print({k:(round(v,4) if isinstance(v,float) else v) for k,v in r.items()},flush=True)
json.dump(out,open('/mnt/project-files/model/s52/r392/ana/i1i4.json','w'))
