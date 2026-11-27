# S0 (Σ|z|² початкового стану) для продовжень (392)/(393): ck вихідних прогонів r373/r374 видалено, S0 детермінований зерном.
import sys,numpy as np; sys.argv=['x']; sys.path[:0]=['/mnt/project-files/model/s52/r351','/mnt/project-files/model/s52']
import rq351f as Q
for sd in range(1,9):
    s,f,mod,z,psi=Q.run(L=16,seed=sd,check=1,s0=3.0,ell=4.0)
    S0=float((np.abs(z)**2).sum()); np.savez(f'/mnt/project-files/model/s52/r392/s0ck/s{sd}_ck.npz',S0=S0); print(sd,S0)
