"""rq458 (22.11, (458) етап Б): справжнє поле PFC вирощує ґратку, Матриця живе на ній (постулат автора 22.11: ґратка росте лише під живою Матрицею).
Граф — Кун L16 (як (457)); вузол v має місце x_v = (ijk+0,5)·LB/16 у ящику PFC 13 комірок ОЦК (104^3).
Затвор: eps(x)=0,25 у кристалі (c>0,5) і в зоні біля вузлів ЖИВОЇ Матриці (вузол ґратки, середнє |z|^2 його живих ниток > rd=1); деінде EOFF.
Вузол хаосу стає ґраткою, коли c(x_v)>0,5 і має сусіда-ґратку (нові нитки — як (457)).
Годинники: на кожен крок зв'язку dtc=0,5 часу Матриці — r·dtc часу PFC (dtP=0,5). Якщо зони росту поза кристалом нема — PFC заморожено (наближення, записано).
python3 rq458.py r seed [Tt] -> runs/r{r}_s{seed}.json/.npz; ck кожні 5 часу Матриці. env H, EOFF, NOWORLD."""
import os, sys, json, time, numpy as np
os.environ["NB"]="0"; os.environ.setdefault("DELTA","1e-6")
R="/mnt/project-files/model/s52"; sys.path[:0]=[R+"/r458",R+"/r396",R+"/r351",R]
import fastpatch; fastpatch.install()
import rq351f as Q, pfcB as P
rc=float(sys.argv[1]); seed=int(sys.argv[2]); Tt=float(sys.argv[3]) if len(sys.argv)>3 else 200.0
s0=3.0; h=float(os.environ.get('H','0.36')); EOFF=float(os.environ.get('EOFF','0.20')); ac=0.1; L=16; rd=1.0; R0=3.5; dt=0.005; dtc=0.5; dtP=0.5
SIGZ=float(os.environ.get('SIGZ','1.0'))*P.LB/16    # ширина зони росту біля живого вузла: 1 крок Куна
NOW=os.environ.get('NOWORLD','0')=='1'
AMODE=os.environ.get('ALIVE_M','0'); AM=AMODE!='0'
RPK=2.6577   # (454) вершина стійкої гілки G при h0,36 (rho_pk(psiM)); варіант А, обраний автором 22.11   # (458) виправлення 22.11: живий = Матриця (ρ>rd і ψ<0), світ ріст не відкриває
s,f,mod,z,psi=Q.run(L=L,seed=seed,s0=s0,h=h,ac=ac,check=1)
r_=np.roots([1,0,-1,-h]); r_=r_[np.isreal(r_)].real; psiM=r_.min()
ta,hd=f.tail,f.head; n,m=f.n,f.m; F0=s.F; corr=s.lc*s0/(4*s.wp**2)
nb=[[] for _ in range(n)]; lk=[[] for _ in range(n)]
for e,(p,q) in enumerate(zip(ta,hd)): nb[p].append(q); nb[q].append(p); lk[p].append(e); lk[q].append(e)
rng=np.random.default_rng(500+seed)
os.makedirs("runs",exist_ok=True)
name=f"runs/r{rc:g}_h{h:g}_e{EOFF:g}_s{seed}"+("_nw" if NOW else "")+(("_am" if AMODE=="1" else "_A") if AM else ""); ckf=name+"_ck.npz"
cen=f.ijk[rng.integers(n)]
dI=(f.ijk-cen+L//2)%L-L//2; rr=np.linalg.norm(dI,axis=1)
gidx=(np.floor((f.ijk+0.5)*P.NG/L)).astype(int)%P.NG          # вузол -> комірка сітки PFC
if os.path.exists(ckf):
    c=np.load(ckf,allow_pickle=True); z=c["z"]; psi=c["psi"]; act=c["act"]; t=float(c["t"]); S0=float(c["S0"]); hist=list(c["hist"]); phi=c["phi"].astype(float); tP=float(c["tP"]); print("resume",t,flush=True)
else:
    act=rr<=R0; ae=act[ta]&act[hd]
    z=np.where(ae,np.sqrt(s0)*(1+0.01*(rng.standard_normal(m)+1j*rng.standard_normal(m))),0j)
    psi=psiM+0.01*rng.standard_normal(n)*act+0*psi; psi[~act]=psiM
    S0=s0*ae.sum()+0.05*(~ae).sum(); t=0.0; hist=[]; tP=0.0
    phi=P.seedball(rng,(cen+0.5)*P.LB/L,R0*P.LB/L)
    # дозрівання зародка без Матриці: поле до c>0,5 у кулі (≈ t50 PFC, eps0,25 лише в кулі)
    x=np.arange(P.NG)*P.DX; X,Y,Z=np.meshgrid(x,x,x,indexing='ij'); cc=(cen+0.5)*P.LB/L
    ball=np.sqrt(sum((((V-ci+P.LB/2)%P.LB)-P.LB/2)**2 for V,ci in zip((X,Y,Z),cc)))<R0*P.LB/L
    for _ in range(20):
        phi=P.step(phi,np.where(ball,P.EPS0,EOFF),dtP,10); tP+=5
        if (P.cryst(phi)[ball]>0.5).mean()>0.9: break
    del X,Y,Z
def setup():
    global ae, ninact
    ae=act[ta]&act[hd]; ninact=np.bincount(ta,~ae,n)+np.bincount(hd,~ae,n)
setup()
def F(zz,pp):
    dz,fp=F0(zz,pp)
    A2=np.abs(zz)**2; rho=np.maximum(A2,0.05); D=(rho.sum()-s.S0)*(1.0/max(ae.sum(),1)-1.0/m)
    dz=np.where(ae,dz-D/(2*rho)*zz,0); fp=fp+corr*ninact; fp=np.where(act,fp,0.0)
    if NOW: fp=0*fp
    return dz,fp
s.F=F; s.S0=S0; s.m0=s0
nP=max(1,int(round(rc*dtc/dtP))); frozen=0; cfield=P.cryst(phi)
def alive_nodes():
    A2=np.abs(z)**2; num=np.bincount(ta,A2*ae,n)+np.bincount(hd,A2*ae,n); den=np.bincount(ta,ae,n)+np.bincount(hd,ae,n)
    return act&(den>0)&(num/np.maximum(den,1)>rd)&((psi<0) if AM else True)
def couple():
    global phi, cfield, S0, tP, frozen
    cr=cfield>0.5
    isedge=act&np.array([any(not act[w] for w in nb[v]) for v in range(n)])
    if AMODE=='A':   # варіант А: Матриця за краєм (сусіди на крок углиб, psi<0) на стійкій гілці: середня rho вузла > RPK
        A2=np.abs(z)**2; num=np.bincount(ta,A2*ae,n)+np.bincount(hd,A2*ae,n); den=np.bincount(ta,ae,n)+np.bincount(hd,ae,n); rn=num/np.maximum(den,1)
        edge=np.zeros(n,bool)
        for v in np.flatnonzero(isedge):
            inn=[w for w in nb[v] if act[w] and not isedge[w] and psi[w]<0]
            edge[v]=bool(inn) and rn[inn].mean()>RPK
    else:
        edge=alive_nodes()&isedge
    # зона росту: біля живих крайових вузлів; всередині кристал і так тримає 0,25
    zone=(P.splat(gidx[edge],SIGZ)>0.3) if edge.any() else np.zeros(cr.shape,bool)
    if (zone&~cr).any():
        phi=P.step(phi,np.where(cr|zone,P.EPS0,EOFF),dtP,nP); tP+=nP*dtP; cfield=P.cryst(phi)
    else: frozen+=1
    cv=cfield[gidx[:,0],gidx[:,1],gidx[:,2]]>0.5
    new=[v for v in np.flatnonzero(~act&cv) if any(act[w] for w in nb[v])]
    for v in new:
        act[v]=True
        for e,w in zip(lk[v],nb[v]):
            if act[w] and not (act[ta[e]] and act[hd[e]] and abs(z[e])>0):
                ne=[x for x in lk[w] if ae[x]]; ph=np.angle(z[ne].mean()) if ne else 0.0
                z[e]=np.sqrt(s0)*np.exp(1j*(ph+0.01*rng.standard_normal())); S0+=s0-0.05
        psi[v]=psiM
    if new: setup(); s.S0=S0
    return int(edge.sum())
def stats(nedge):
    W=(psi>0)&act; cnt=int(act.sum()); A2=np.abs(z)**2; rmax_l=float(rr[act].max()); Mm=ae&~(W[ta]|W[hd])
    return dict(t=round(t,2),tP=round(tP,1),nlat=cnt,nW=int(W.sum()),cfrac=round(float((cfield>0.5).mean()),4),nedge=nedge,frozen=frozen,
                rmax=round(rmax_l,2),rhoM=round(float(A2[Mm].mean()) if Mm.any() else 0,3),
                rhoW=round(float(A2[ae&W[ta]&W[hd]].mean()) if (ae&W[ta]&W[hd]).any() else 0,3),S0=round(S0,1))
step=int(round(dtc/dt)); i=int(round(t/dt)); nedge=0; t0=time.time()
while t<Tt-1e-9:
    z,psi=s.step(z,psi,dt); i+=1; t=i*dt
    z=np.where(ae,z,0j); psi=np.where(act,psi,psiM)
    if i%step==0: nedge=couple()
    if i%int(round(1/dt))==0:
        st=stats(nedge); hist.append(st)
        if str(int(round(t))) in os.environ.get('SNAPS','').split(','): np.savez(name+f'_snap{int(round(t))}.npz',z=z,psi=psi,act=act)
        if i%int(round(5/dt))==0: print(st, round(time.time()-t0), flush=True)
        if not np.isfinite(psi).all(): break
    if i%int(round(5/dt))==0:
        np.savez(ckf+".tmp.npz",z=z,psi=psi,act=act,t=t,S0=S0,hist=np.array(hist,dtype=object),phi=phi.astype(np.float32),tP=tP); os.replace(ckf+".tmp.npz",ckf)
np.savez(name+".npz",z=z,psi=psi,act=act,phi=phi.astype(np.float32))
json.dump(dict(r=rc,seed=seed,s0=s0,h=h,eoff=EOFF,ac=ac,rd=rd,nP=nP,hist=hist,cen=cen.tolist()),open(name+".json","w"))
if os.path.exists(ckf): os.remove(ckf)
print("done",name)
