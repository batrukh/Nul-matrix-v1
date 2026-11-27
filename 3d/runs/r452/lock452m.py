"""(452Г) як lock452L.py, але частку замкнених петель рахує окремо для петель Матриці (усі 4 вузли ψ<0 на початку і в кінці вікна), світу (усі 4 ψ>0) і фронту (решта). python3 lock452m.py ac seed L"""
import sys, json, numpy as np
R="/mnt/project-files/model/s52"; sys.path[:0]=[R+"/r447",R+"/r396",R+"/r351",R]
import fastpatch; fastpatch.install()
import rq351f as Q
ac=float(sys.argv[1]); seed=int(sys.argv[2]); L=int(sys.argv[3])
s,f,mod,z,psi = Q.run(L=L, seed=seed, s0=3.0, h=0.36, ac=ac, check=1)
rng=np.random.default_rng(100+seed); A=np.sqrt(3.0)
z=A*np.exp(1j*1e-3*rng.standard_normal(f.m)); r_=np.roots([1,0,-1,-0.36]); psi=np.full(f.n,r_[np.isreal(r_)].real.min())
s.S0=float((np.abs(z)**2).sum()); s.m0=s.S0/f.m
sig=np.abs(np.array(mod.S).sum(0)); QV=np.array(mod.QV)
dt=0.005; snaps={}; psis={}
ph_prev=np.angle(mod.loop(z)[1]); acc=np.zeros(len(sig))
for i in range(1,int(25/dt)+1):
    z,psi=s.step(z,psi,dt); t=i*dt
    ph=np.angle(mod.loop(z)[1]); acc+=np.angle(np.exp(1j*(ph-ph_prev))); ph_prev=ph
    if i%int(5/dt)==0: snaps[round(t)]=acc.copy(); psis[round(t)]=psi.copy()
out=dict(ac=ac,seed=seed,L=L,psimax=float(psi.max()),world_nodes_t25=int((psi>0).sum()))
for (t0,t1) in ((10,15),(20,25)):
    dd=np.abs(snaps[t1]-snaps[t0]); p0=psis[t0][QV]; p1=psis[t1][QV]
    cls={"M":(p0<0).all(1)&(p1<0).all(1), "W":(p0>0).all(1)&(p1>0).all(1)}; cls["F"]=~(cls["M"]|cls["W"])
    for c,mc in cls.items():
        for sv in (0,2):
            m=mc&(sig==sv); out[f"n_{c}{sv}_{t0}"]=int(m.sum())
            out[f"lock_{c}{sv}_{t0}"]=float((dd[m]<0.5).mean()) if m.any() else None
Ph=np.angle(mod.loop(z)[1]); pM=(psi[QV]<0).all(1)
out["twist_M2"]=float((1-np.cos(Ph[pM&(sig==2)])).mean()) if (pM&(sig==2)).any() else None
print(json.dumps(out)); json.dump(out,open(f"lock_a{ac:g}_s{seed}_L{L}.json","w"))
