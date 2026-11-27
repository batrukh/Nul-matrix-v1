"""Рис. 282: (а) |z|² Матриці від відстані до межі ±α у доменах без світу (391)+(385)+(387) проти тору без меж; (б) острів: частка домену проти |z|² Матриці далеко від меж."""
import sys, glob, json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
sys.argv=['x']; sys.path[:0]=['/mnt/project-files/model/s52/r351','/mnt/project-files/model/s52']
import rq351f as Q
s,f,mod,z0,p0=Q.run(L=16,check=1); T,H=f.tail,f.head; L=16
xt=f.ijk[T,0].astype(float); d=(f.ijk[H,0]-xt+L/2)%L-L/2; xm=(xt+d/2)%L
o=json.load(open('i1i4.json')); R='/mnt/project-files/model/s52/'
dirs={6:'r391/runs/sp6_seed{}',4:'r391/runs/sp4_seed{}',8:'r385/runs/a0.2_seed{}',12:'r387/runs/seed{}'}
bins=np.arange(0,8.5,0.5); prof=[]
for r in o:
    if r['split']==16 or r['big']>=50: continue
    c=np.load(glob.glob(R+dirs[r['split']].format(r['seed'])+'/t3_*_ell4.npz')[0]); A=np.abs(c['z'])**2; p=c['psi']; mat=(p[T]<0)&(p[H]<0)
    sp=r['split']; dw=np.minimum(np.abs(((xm-(sp-.5))+8)%16-8),np.abs(((xm-15.5)+8)%16-8))
    prof.append([A[mat&(dw>=a)&(dw<b)].mean() if (mat&(dw>=a)&(dw<b)).any() else np.nan for a,b in zip(bins[:-1],bins[1:])])
prof=np.array(prof); m=np.nanmean(prof,0); sd=np.nanstd(prof,0)
uni=[r['Amat'] for r in o if r['split']==16 and r['big']<50]
fig,ax=plt.subplots(1,2,figsize=(10.5,4.2)); bc=(bins[:-1]+bins[1:])/2
ax[0].errorbar(bc,m,yerr=sd,fmt='o-',color='#d62728',capsize=2,label=f'домени без світу ({len(prof)} прогонів)')
ax[0].axhline(np.mean(uni),color='#1f77b4',ls='--',label=f'тор без меж, без світу ({len(uni)}): {np.mean(uni):.2f}')
ax[0].axhline(3.0,color='gray',ls=':',lw=.8,label='s0 = 3,0')
ax[0].set_xlim(0,6.5); ax[0].set_xlabel('відстань до межі ±α, шарів'); ax[0].set_ylabel('|z|² Матриці'); ax[0].legend(fontsize=7)
ax[0].set_title('(а) межа збіднена, глибина домену збагачена',fontsize=9)
isl=[(w['world']/(w['width']*256),w['Afar'],r['split']) for r in o if r['split']<16 for k in 'AB' for w in [r[k]] if w['world']>0]
x=[a for a,_,_ in isl]; y=[b for _,b,_ in isl]
ax[1].plot(x,y,'o',color='#d62728',label='острови в доменах (391)+(385)+(387)')
ax[1].plot([.282,.203],[2.693,2.83],'s',color='#1f77b4',label='застряглі без меж (390)')
ax[1].axhspan(np.nanmean(prof[:,6:].ravel())-.1,np.nanmean(prof[:,6:].ravel())+.1,color='#d62728',alpha=.12,label='глибина домену без світу')
ax[1].axhline(np.mean(uni),color='#1f77b4',ls='--',lw=.8)
rho,pv=spearmanr(x+[.282,.203],y+[2.693,2.83])
ax[1].set_xlabel('острів, частка свого домену (або ґратки)'); ax[1].set_ylabel('|z|² Матриці далеко від меж')
ax[1].set_title(f'(б) великі острови виснажують Матрицю, малі — ні\n(Спірмен ρ = {rho:.2f}, p = {pv:.2f}, 8 островів)',fontsize=9); ax[1].legend(fontsize=7)
fig.suptitle('(394) І-1(а) та І-4 на готових станах t300: Кун L16, α 0,2, s0 3,0',fontsize=10); fig.tight_layout()
fig.savefig('/mnt/project-files/figures/fig282_3d_zapas_matryci.png',dpi=130)
print('profile',np.round(m,2).tolist()); print('uni',np.round(uni,3).tolist()); print('isl',isl,'rho',rho,pv)
