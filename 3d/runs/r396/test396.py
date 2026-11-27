"""(396) перевірка fastpatch: (1) сила F нова проти старої на кількох станах (зокрема зі світом і з α-масивом ±α);
(2) траєкторія 2 од. часу (400 кроків RK4) — максимальна відносна різниця z, ψ; (3) швидкість кроку."""
import os; os.environ["NB"]="0"; os.environ.setdefault("DELTA","1e-6")
import sys, time, numpy as np; sys.argv=['x']; sys.path[:0]=['/mnt/project-files/model/s52/r396','/mnt/project-files/model/s52/r351','/mnt/project-files/model/s52']
import local60 as LC; OLD = LC.Loc.F
import fastpatch; fastpatch.install(); NEW = LC.Loc.F
import rq351f as Q
def mk(ac, seed=3):
    s,f,mod,z,psi=Q.run(L=16,seed=seed,check=1,s0=3.0,ac=ac); s.S0=float((np.abs(z)**2).sum()); s.m0=s.S0/f.m; return s,f,z,psi
out=[]
for name,ac,arr,st in (('α0,1 старт',0.1,False,None),('±α0,2 масив',0.2,True,None),('світ r374 α0,1',0.1,False,'/mnt/project-files/model/s52/r374/a0.1_seed3/t3_s3_L16_s3_ell4.npz')):
    s,f,z,psi=mk(ac)
    if arr: xm=f.ijk[f.tail,0]; s.cc=1+1j*np.where(xm<8,ac,-ac)
    if st: d=np.load(st); z=d['z']; psi=d['psi']
    a=OLD(s,z,psi); b=NEW(s,z,psi)
    rz=np.abs(a[0]-b[0]).max()/np.abs(a[0]).max(); rp=np.abs(a[1]-b[1]).max()/np.abs(a[1]).max()
    print(name,'F: відн. різниця dz',rz,'dψ',rp, flush=True)
    # траєкторія 400 кроків
    za,pa=z.copy(),psi.copy(); zb,pb=z.copy(),psi.copy()
    LC.Loc.F=OLD; t0=time.time()
    for _ in range(400): za,pa=s.step(za,pa,0.005)
    t1=time.time(); LC.Loc.F=NEW
    for _ in range(400): zb,pb=s.step(zb,pb,0.005)
    t2=time.time()
    print(name,'T2: max|Δz|',np.abs(za-zb).max(),'max|Δψ|',np.abs(pa-pb).max(),'старий мс/крок',round((t1-t0)/0.4,1),'новий',round((t2-t1)/0.4,1),flush=True)
