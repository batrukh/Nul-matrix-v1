"""(447) проба 21.11 ~00:30 UTC (попередньо, 1 зерно): шумовий відпал (421Б–В, T 0,15) скляної ґратки S1 у ящику 13 комірок. Блоки t500 шуму + t200 гарту → CNA, q6. ck кожен блок."""
import sys,os,json,numpy as np,time
R="/mnt/project-files/model/s52"; sys.path[:0]=[R+"/r421",R+"/r447"]
from pfc import kgrid,step,step_noise,energy,A_BCC; import classify as C; from pfc447 import peaks,CELLS,NG
seed=int(sys.argv[1]); T=float(sys.argv[2]); nblk=int(sys.argv[3])
Lb=CELLS*A_BCC; Ls=(Lb,)*3; k2=kgrid((NG,)*3,Ls); pb=-0.30
ck=f"probe/ann_S{seed}_T{T}_ck.npz"; out=f"probe/ann_S{seed}_T{T}.json"
if os.path.exists(ck): c=np.load(ck,allow_pickle=True); psi=c['psi'].astype(float); b0=int(c['b']); hist=list(c['hist'])
else:
    c=np.load(f"runs/lat_S_s{seed}_ck.npz") if os.path.exists(f"runs/lat_S_s{seed}_ck.npz") else None
    psi=np.load(f"probe/psi_S{seed}.npy").astype(float); b0=0; hist=[]
rng=np.random.default_rng(300+seed+b0)
for b in range(b0,nblk):
    t0=time.time(); psi=step_noise(psi,k2,0.3,0.5,1000,T,Ls,rng); q=step(psi,k2,0.3,0.5,400)
    P=peaks(q,Lb,pb); fr,_=C.cna(P,np.array(Ls)); _,q6=C.qbar(P,np.array(Ls),avg=True)
    hist.append(dict(t=500*(b+1),F=energy(q,k2,0.3),npk=len(P),q6=float(q6.mean()),**fr,sec=time.time()-t0)); print(hist[-1],flush=True)
    np.savez(ck+".tmp.npz",psi=psi.astype(np.float32),b=b+1,hist=np.array(hist,dtype=object)); os.replace(ck+".tmp.npz",ck)
    json.dump(hist,open(out,'w'),indent=1)
np.savez(f"probe/ann_S{seed}_T{T}_final.npz",P=P,box=Lb)
