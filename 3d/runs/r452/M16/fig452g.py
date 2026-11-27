import json,glob,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
A=[0,0.05,0.1,0.15,0.2,0.25,0.3,0.4]
def m(a,c,t0,minn=200):
    v=[json.load(open(f)) for f in glob.glob(f'lock_a{a:g}_s*_L16.json')]
    x=[d[f'lock_{c}2_{t0}'] for d in v if d[f'n_{c}2_{t0}']>=minn and d[f'lock_{c}2_{t0}'] is not None]
    return (np.mean(x),np.std(x,ddof=1)/np.sqrt(len(x))) if len(x)>1 else (np.nan,np.nan)
fig,ax=plt.subplots(1,2,figsize=(9,3.6),sharey=True)
y=[m(a,'M',10) for a in A]; ax[0].errorbar(A,[q[0] for q in y],[q[1] for q in y],fmt='o-',color='C0')
ax[0].set_title('t 10–15: світу ще нема — петлі Матриці',fontsize=9)
for c,lab,col in (('M','Матриця','C0'),('F','фронт','C1'),('W','світ','C2')):
    y=[m(a,c,20) for a in A]; ax[1].errorbar(A,[q[0] for q in y],[q[1] for q in y],fmt='o-',color=col,label=lab)
ax[1].set_title('t 20–25: світ уже є (α 0,1–0,2)',fontsize=9); ax[1].legend(fontsize=8)
for a_ in ax: a_.set_xlabel('α'); a_.axvspan(0.05,0.2,color='g',alpha=0.08)
ax[0].set_ylabel('частка замкнених петель σ=2'); ax[0].set_ylim(0.2,1.03)
fig.suptitle('(452Г) Де дрейфують петлі, Кун L16, h 0,36, 8 зерен'); fig.tight_layout()
fig.savefig('/mnt/project-files/figures/fig452g.png',dpi=130)
