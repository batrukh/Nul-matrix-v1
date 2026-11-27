import sys, numpy as np
R="/mnt/project-files/model/s52"; sys.path[:0]=[R+"/r447",R+"/r396",R+"/r351",R]
import fastpatch; fastpatch.install()
import rq351f as Q
s0=float(sys.argv[1]); h=0.36
s,f,mod,z,psi=Q.run(L=4,seed=1,s0=s0,h=h,ac=0.0,check=1)
r_=np.roots([1,0,-1,-h]); psiM=r_[np.isreal(r_)].real.min()
s.S0=s0*f.m; s.m0=s0; A=np.sqrt(s0)
def F(dr,dp):  # radial perturbation of amplitudes dr (on links) and psi
    z=(A+dr)+0j; p=psiM+dp; dz,dpp=s.F(z,p); return dz.real, dpp
rng=np.random.default_rng(0); v=rng.standard_normal(f.m); v-=v.mean(); e=1e-6
r1,p1=F(e*v,np.zeros(f.n)); r0,p0=F(-e*v,np.zeros(f.n))
dA=(r1-r0)/(2*e); print("dA/dA response ratio (proj on v)", (dA@v)/(v@v), "psi resp norm", np.linalg.norm((p1-p0)/(2*e)))
w=rng.standard_normal(f.n); w-=w.mean()
r1,p1=F(np.zeros(f.m),e*w); r0,p0=F(np.zeros(f.m),-e*w)
print("A resp to psi norm",np.linalg.norm((r1-r0)/(2*e)),"psi self proj",((p1-p0)/(2*e))@w/(w@w))
print("K info", getattr(s.K,'Kk_half',None) is not None)
