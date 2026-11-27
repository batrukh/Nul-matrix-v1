"""(470Г) лінійний відгук однорідної Матриці на джерело pm470 без вільних параметрів: δx = −μ J⁻¹ f (якобіан повної моделі, радіальний сектор, α 0).
python3 lin470g.py s0 L seed  → lin_s{s0}_L{L}_s{seed}.json (профіль δρ(r) як pm470.prof, відносно середнього по ящику) + порівняння з даними r470/runs."""
import sys, os, json, time, numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
R="/mnt/project-files/model/s52"; sys.path[:0]=[R+"/r447",R+"/r396",R+"/r351",R]
import fastpatch; fastpatch.install()
import rq351f as Q
s0=float(sys.argv[1]); L=int(sys.argv[2]); seed=int(sys.argv[3]); H=0.36; MU=float(sys.argv[4]) if len(sys.argv)>4 else 0.2; R0=1.0
s,f,mod,z_,psi_=Q.run(L=L,seed=seed,s0=s0,h=H,ac=0.0,check=1)
Sdat=float((np.abs(z_)**2).sum())          # S0 прогону (з шумом посадки)
r_=np.roots([1,0,-1,-H]); psiM=r_[np.isreal(r_)].real.min(); A=np.sqrt(s0)
s.S0=s0*f.m; s.m0=s0; m,n=f.m,f.n
rng=np.random.default_rng(500+seed); c0=int(rng.integers(f.n))
dI=(f.ijk-f.ijk[c0]+L//2)%L-L//2; X=dI.astype(float); rn=np.linalg.norm(X,axis=1)
mid=(X[f.tail]+((X[f.head]-X[f.tail]+L//2)%L-L//2)/2); re=np.linalg.norm(mid,axis=1); ball=re<=R0+1e-9
F0=s.F
def Fsrc(z,psi,mu):
    dz,fp=F0(z,psi)
    if mu==0: return dz,fp
    rho=np.maximum(np.abs(z)**2,0.05); ex=np.where(ball,mu,0.0)*z; src=2*(z.conjugate()*ex).real
    ks=fastpatch._krfft(s,src); return dz+ex-ks/(2*rho)*z,fp
def Fr(x,mu=0.0):
    dz,dp=Fsrc((A+x[:m])+0j,psiM+x[m:],mu); return np.concatenate([dz.real,dp])
x0=np.zeros(m+n); base=Fr(x0); print("|F(hom)|",np.abs(base).max(),flush=True)
e=1e-6
fvec=(Fr(x0,1e-4)-Fr(x0,-1e-4))/(2e-4)       # ∂F/∂μ
nm=[0]
def mv(v):
    nm[0]+=1; return (Fr(e*v)-Fr(-e*v))/(2*e)
Op=LinearOperator((m+n,m+n),matvec=mv,dtype=float)
t0=time.time(); it=[0]
def cb(r):
    it[0]+=1
    if it[0]%200==0: print('it',it[0],'res',r,round(time.time()-t0),flush=True)
dx,info=gmres(Op,-MU*fvec,rtol=1e-7,atol=0,restart=300,maxiter=3,callback=cb,callback_type='pr_norm')
res=np.linalg.norm(mv(dx)+MU*fvec)/np.linalg.norm(MU*fvec); print('|b|',np.linalg.norm(MU*fvec)); print("lgmres info",info,"rel.res",res,"matvecs",nm[0],round(time.time()-t0),"s",flush=True)
drho=2*A*dx[:m]; drho-=drho.mean()       # відносно середнього по ящику
rb=np.round(re*2)/2; P={}
for r in np.unique(rb):
    if r>L/2: continue
    sel=rb==r; P[f"{r:g}"]=[float(drho[sel].mean()),int(sel.sum())]
out=dict(s0=s0,L=L,seed=seed,mu=MU,c0=c0,info=int(info),relres=float(res),prof_drho=P,Sdat_mean=Sdat/m)
fn=f"{R}/r470/runs/s{s0:g}_a0_L{L}_mu{MU:g}_s{seed}.json"
if os.path.exists(fn):
    d=json.load(open(fn)); assert d["c0"]==c0
    cmp=[]
    for k,(v,w) in P.items():
        dd=d["prof_rho"][k][0]-Sdat/m; cmp.append([float(k),dd,v,(dd-v)/v if v!=0 else None])
    out["cmp"]=cmp
    for row in cmp:
        if 1<=row[0]<=L/2: print("r %4.1f  дані %+.5f  лін %+.5f  відн %+.3f"%tuple(row))
json.dump(out,open(f"lin_s{s0:g}_L{L}_mu{MU:g}_s{seed}.json","w"),indent=0)
