"""(453) найбільші власні числа радіально-ψ сектора (α 0, плаский однорідний стан) великої ґратки Куна: ARPACK + похідна за напрямком. python3 eig453.py s0 h L"""
import sys, json, time, numpy as np
from scipy.sparse.linalg import LinearOperator, eigs
R="/mnt/project-files/model/s52"; sys.path[:0]=[R+"/r447",R+"/r396",R+"/r351",R]
import fastpatch; fastpatch.install()
import rq351f as Q
s0=float(sys.argv[1]); h=float(sys.argv[2]); L=int(sys.argv[3])
s,f,mod,z,psi=Q.run(L=L,seed=1,s0=s0,h=h,ac=0.0,check=1)
r_=np.roots([1,0,-1,-h]); psiM=r_[np.isreal(r_)].real.min(); A=np.sqrt(s0)
s.S0=s0*f.m; s.m0=s0; m,n=f.m,f.n
def Fr(x):
    dz,dp=s.F((A+x[:m])+0j, psiM+x[m:]); return np.concatenate([dz.real,dp])
e=1e-6; nm=[0]
def mv(v):
    nm[0]+=1; return (Fr(e*v)-Fr(-e*v))/(2*e)
t0=time.time(); Op=LinearOperator((m+n,m+n),matvec=mv,dtype=float)
ev,V=eigs(Op,k=6,which='LR',tol=1e-7,maxiter=3000)
o=np.argsort(-ev.real); ev=ev[o]; V=V[:,o]
# що це за мода: частка у ψ і довжина хвилі (через автокореляцію ψ-частини)
info=[]
for j in range(len(ev)):
    v=np.real(V[:,j]); fpsi=float((v[m:]**2).sum()/(v**2).sum())
    P=v[m:].reshape(L,L,L) if n==L**3 else None
    kdom=None
    if P is not None:
        S=np.abs(np.fft.fftn(P))**2; S[0,0,0]=0; idx=np.unravel_index(np.argmax(S),S.shape); kk=[(i if i<=L//2 else i-L) for i in idx]; kdom=[int(x) for x in kk]
    info.append(dict(re=float(ev[j].real),im=float(ev[j].imag),fpsi=round(fpsi,3),kdom=kdom))
out=dict(s0=s0,h=h,L=L,matvecs=nm[0],sec=round(time.time()-t0,1),modes=info)
print(json.dumps(out)); json.dump(out,open(f"eig_s{s0:g}_h{h:g}_L{L}.json","w"))
