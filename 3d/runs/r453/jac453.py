"""(453) якобіан повної моделі в однорідному стані (α 0): найбільше дійсне власне число без нульових мод. python3 jac453.py s0 h [L]"""
import sys, json, numpy as np
R="/mnt/project-files/model/s52"; sys.path[:0]=[R+"/r447",R+"/r396",R+"/r351",R]
import fastpatch; fastpatch.install()
import rq351f as Q
s0=float(sys.argv[1]); h=float(sys.argv[2]); L=int(sys.argv[3]) if len(sys.argv)>3 else 4
s,f,mod,z,psi=Q.run(L=L,seed=1,s0=s0,h=h,ac=0.0,check=1)
r_=np.roots([1,0,-1,-h]); psiM=r_[np.isreal(r_)].real.min()
z0=np.full(f.m,np.sqrt(s0)+0j); p0=np.full(f.n,psiM); s.S0=s0*f.m; s.m0=s0
def Fv(y):
    z=y[:f.m]+1j*y[f.m:2*f.m]; p=y[2*f.m:]; dz,dp=s.F(z,p); return np.concatenate([dz.real,dz.imag,dp])
y0=np.concatenate([z0.real,z0.imag,p0]); F0=Fv(y0); N=len(y0); J=np.empty((N,N)); e=1e-6
for i in range(N):
    y=y0.copy(); y[i]+=e; yb=y0.copy(); yb[i]-=e; J[:,i]=(Fv(y)-Fv(yb))/(2*e)
ev=np.linalg.eigvals(J); ev=ev[np.argsort(-ev.real)]
nz=ev[np.abs(ev)>1e-6]
out=dict(s0=s0,h=h,L=L,F0max=float(np.abs(F0).max()),lmax=float(nz[0].real),top=[[round(x.real,5),round(x.imag,5)] for x in ev[:6]],nzero=int((np.abs(ev)<=1e-6).sum()))
print(json.dumps(out)); json.dump(out,open(f"jac_s{s0:g}_h{h:g}_L{L}.json","w"))
