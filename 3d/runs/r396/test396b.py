"""(396б) чи розбіжність на стані зі світом — властивість самої динаміки: старий код проти старого з збуренням 1e-15, і новий проти старого, по кроках."""
import os; os.environ["NB"]="0"; os.environ.setdefault("DELTA","1e-6")
import sys, numpy as np; sys.argv=['x']; sys.path[:0]=['/mnt/project-files/model/s52/r396','/mnt/project-files/model/s52/r351','/mnt/project-files/model/s52']
import local60 as LC; OLD = LC.Loc.F
import fastpatch; fastpatch.install(); NEW = LC.Loc.F
import rq351f as Q
s,f,mod,z,psi=Q.run(L=16,seed=3,check=1,s0=3.0,ac=0.1); s.S0=float((np.abs(z)**2).sum()); s.m0=s.S0/f.m
d=np.load('/mnt/project-files/model/s52/r374/a0.1_seed3/t3_s3_L16_s3_ell4.npz'); z=d['z']; psi=d['psi']
print('мертвих ниток |z|<1e-3:',int((np.abs(z)<1e-3).sum()),'з',f.m)
rng=np.random.default_rng(0); zp=z*(1+1e-15*rng.standard_normal(f.m))
A=[z.copy(),psi.copy()]; B=[zp.copy(),psi.copy()]; C=[z.copy(),psi.copy()]
for i in range(1,401):
    LC.Loc.F=OLD; A=list(s.step(*A,0.005)); B=list(s.step(*B,0.005)); LC.Loc.F=NEW; C=list(s.step(*C,0.005))
    if i in (1,10,50,100,200,400):
        live=np.abs(A[0])>0.1
        print(i,'старий±1e-15: Δz',f"{np.abs(A[0]-B[0]).max():.1e}",'живі',f"{np.abs(A[0]-B[0])[live].max():.1e}",' новий−старий: Δz',f"{np.abs(A[0]-C[0]).max():.1e}",'живі',f"{np.abs(A[0]-C[0])[live].max():.1e}",'Δψ',f"{np.abs(A[1]-C[1]).max():.1e}",flush=True)
