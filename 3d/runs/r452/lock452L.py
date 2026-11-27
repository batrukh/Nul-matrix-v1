"""(452) числова перевірка: частка замкнених петель від α (Кун L8, плаский старт). python3 lock452L.py ac seed L (24.11, (452В))"""
import sys, json, numpy as np
R="/mnt/project-files/model/s52"; sys.path[:0]=[R+"/r447",R+"/r396",R+"/r351",R]
import fastpatch; fastpatch.install()
import rq351f as Q
ac=float(sys.argv[1]); seed=int(sys.argv[2]); L=int(sys.argv[3]) if len(sys.argv)>3 else 8
s,f,mod,z,psi = Q.run(L=L, seed=seed, s0=3.0, h=0.36, ac=ac, check=1)
rng=np.random.default_rng(100+seed); A=np.sqrt(3.0)
z=A*np.exp(1j*1e-3*rng.standard_normal(f.m)); r_=np.roots([1,0,-1,-0.36]); psi=np.full(f.n,r_[np.isreal(r_)].real.min())
s.S0=float((np.abs(z)**2).sum()); s.m0=s.S0/f.m
sig=np.array(mod.S).sum(0)
def Phi(z): _,P=mod.loop(z); return np.unwrap(np.angle(P))
dt=0.005; t=0; snaps={}; tot={}
ph_prev=np.angle(mod.loop(z)[1]); acc=np.zeros(len(sig))
for i in range(1,int(25/dt)+1):
    z,psi=s.step(z,psi,dt); t=i*dt
    ph=np.angle(mod.loop(z)[1]); d=np.angle(np.exp(1j*(ph-ph_prev))); acc+=d; ph_prev=ph
    if i%int(5/dt)==0: snaps[round(t)]=acc.copy()
out=dict(ac=ac,seed=seed,psimax=float(psi.max()))
for (t0,t1) in ((10,15),(20,25)):
    dd=np.abs(snaps[t1]-snaps[t0])
    for sv in (0,2):
        m=np.abs(sig)==sv; out[f"lock_s{sv}_{t0}"]=float((dd[m]<0.5).mean())
_,Pf=mod.loop(z); Ph=np.angle(Pf)
for sv in (0,2):
    m=np.abs(sig)==sv; out[f"twist_s{sv}"]=float((1-np.cos(Ph[m])).mean()); out[f"absphi_s{sv}"]=float(np.abs(Ph[m]).mean())
print(json.dumps(out)); json.dump(out,open(f"lock_a{ac:g}_s{seed}_L{L}.json","w"))
