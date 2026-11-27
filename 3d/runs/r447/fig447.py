import sys,json,glob,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path[:0]=['../r421','.']; import classify as C, fabpts as F
fig,ax=plt.subplots(1,3,figsize=(14,4.2))
cols={'S':'#1f6fb4','Q':'#d0621f'}
for prot in 'SQ':
    D=[];Q6=[]
    for f in sorted(glob.glob(f'runs/lat_{prot}_s?.npz')):
        d=np.load(f);P=d['P'];L=float(d['box']); fab=F.fromPoints(P,L); D.append(np.asarray(fab.deg)); Q6.append(C.qbar(P,np.array([L]*3),avg=True)[1])
    D=np.concatenate(D);Q6=np.concatenate(Q6)
    b=np.arange(8,22)-0.5; ax[0].hist(D,b,density=True,histtype='step',lw=2,color=cols[prot],label={'S':'повільне охолодження (S)','Q':'різке охолодження (Q)'}[prot])
    ax[1].hist(Q6,np.linspace(0,0.6,61),density=True,histtype='step',lw=2,color=cols[prot],label=prot)
ax[0].axvline(14,color='k',ls='--',label='ОЦК / Кун: усі 14'); ax[0].set_xlabel('число сусідів (спільна грань Вороного)'); ax[0].set_ylabel('частка вузлів'); ax[0].legend(fontsize=8); ax[0].set_title('а) сусіди вузла, 8+8 ґраток')
a=None; rng=np.random.default_rng(1)
L=13*2*np.pi*np.sqrt(2); a=L/13; g=np.arange(13)*a; X=np.array(np.meshgrid(g,g,g,indexing='ij')).reshape(3,-1).T; B=np.vstack([X,X+a/2])
for s,c in ((0.03,'#2a9d4b'),(0.1,'#7a7a7a')):
    q6=C.qbar((B+s*a*rng.standard_normal(B.shape))%L,np.array([L]*3),avg=True)[1]; ax[1].hist(q6,np.linspace(0,0.6,61),density=True,histtype='stepfilled',alpha=0.3,color=c,label='ідеальна ОЦК + шум '+str(s).replace('.',','))
ax[1].set_xlabel('порядок q6 (усереднений)'); ax[1].set_ylabel('густина'); ax[1].legend(fontsize=8); ax[1].set_title('б) локальний порядок: скло, не ОЦК')
h=json.load(open('probe/ann_S8_T0.15.json')); t=[x['t'] for x in h]
ax[2].plot(t,[x['q6'] for x in h],'o-',color='#d0621f',label='q6 (скло Q8)'); ax[2].axhline(0.567,color='k',ls='--',label='ідеальна ОЦК 0,57')
ax[2].set_ylim(0,0.6); ax[2].set_xlabel('час шумового відпалу (T 0,15)'); ax[2].set_ylabel('q6'); ax[2].legend(fontsize=8); ax[2].set_title('в) проба відпалу: ОЦК 0 % до t4000')
fig.suptitle('(447) Ґратка з поля PFC у ящику 13 комірок ОЦК (≈4080 вузлів): скло замість ОЦК',fontsize=11)
fig.tight_layout(); fig.savefig('/mnt/project-files/figures/fig299_3d_pfc_sklo.png',dpi=130)
