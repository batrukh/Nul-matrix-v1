import sys,glob,json,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path[:0]=["/mnt/project-files/model/s52/r447","/mnt/project-files/model/s52"]; import fabpts as F
def degs(files):
    D=[];zm=[]
    for fn in files:
        d=np.load(fn);P=d['P'];L=float(np.atleast_1d(d['box'])[0]); g=np.asarray(F.fromPoints(P,L).deg); D.append(g); zm.append(g.mean())
    return np.concatenate(D),zm
R='/mnt/project-files/model/s52/'
sets={'кристал з повернутого зародка (R)':sorted(glob.glob(R+'r448/runs/R_e0.25_s?.npz')),
      'скло з поля (S, Q)':sorted(glob.glob(R+'r447/runs/lat_[SQ]_s?.npz'))}
rng=np.random.default_rng(1); out={}
fig,ax=plt.subplots(1,2,figsize=(10,4))
bins=np.arange(9.5,22.5)
for (lab,fs),c in zip(sets.items(),['C0','C3']):
    D,zm=degs(fs); out[lab]=zm; ax[0].hist(D,bins,density=True,histtype='step',lw=2,color=c,label=lab)
Pz=[]
for s in range(4):
    P=rng.random((4394,3))*13.0; g=np.asarray(F.fromPoints(P,13.0).deg); Pz.append(g.mean())
    if s==0: ax[0].hist(g,bins,density=True,histtype='step',lw=2,color='gray',label='випадкові точки')
out['випадкові']=Pz
ax[0].axvline(14,c='k',ls=':'); ax[0].set_xlabel('сусідів вузла z (спільна грань Вороного)'); ax[0].set_ylabel('частка вузлів'); ax[0].legend(fontsize=8)
ax[0].set_title('а) розподіл z',fontsize=10)
ys={'кристал (R)':out['кристал з повернутого зародка (R)'],'скло (S, Q)':out['скло з поля (S, Q)'],'випадкові':Pz}
for i,(k,v) in enumerate(ys.items()): ax[1].plot([i]*len(v),v,'o',alpha=.7,color=['C0','C3','gray'][i])
ax[1].axhline(14,c='k',ls=':'); ax[1].set_xticks(range(3)); ax[1].set_xticklabels(list(ys)); ax[1].set_ylabel('середнє z̄ = 2(1 + T/N)')
ax[1].set_title('б) середнє число сусідів: кристал рівно 14',fontsize=10)
plt.tight_layout(); plt.savefig('/mnt/project-files/figures/fig316_voronoi_degree.png',dpi=130); json.dump(out,open('fig316.json','w'))
