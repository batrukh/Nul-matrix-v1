"""fig304: (452) а) частка замкнених петель σ=2 від α: число (Кун L8, 2 зерна) проти середнього поля однієї петлі; б) статичне скручення петель ⟨1−cosΦ⟩ від α."""
import json, glob, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D=[json.load(open(f)) for f in glob.glob('lock_a*.json')]
fig,(a1,a2)=plt.subplots(1,2,figsize=(9,3.8),dpi=150)
for sd,mk in ((1,'o'),(2,'s')):
    r=sorted([d for d in D if d['seed']==sd],key=lambda d:d['ac']); a1.plot([d['ac'] for d in r],[d['lock_s2_20'] for d in r],mk+'-',color='#2a6fb0',ms=4,label=f'число, зерно {sd}')
al=np.linspace(0,0.8,200); kq=np.array([8,12]); 
# середнє поле: частка σ=2-петель з α < ½Σ1/(a k_j); розподіл з Куна (stats.py): 0,52 замкнених при 0,25
mf=np.where(al<0.222,1.0,np.where(al<0.333,np.interp(al,[0.222,0.25,0.333],[1.0,0.52,0.0]),0.0))
a1.plot(al,mf,'--',color='#c0392b',label='середнє поле однієї петлі (передбачення)')
a1.axvspan(0.18,0.35,color='#f9eeda',alpha=0.7,label='очікуваний перехід (критерій)')
a1.set_xlabel('α'); a1.set_ylabel('частка замкнених петель σ=2'); a1.set_ylim(-0.03,1.05); a1.legend(fontsize=7); a1.set_title('а) замикання петель',fontsize=9)
r=sorted([d for d in D if d['seed']==1 and 'twist_s2' in d],key=lambda d:d['ac'])
x=np.array([d['ac'] for d in r]); t0=np.array([d['twist_s0'] for d in r]); t2=np.array([d['twist_s2'] for d in r]); tm=2/3*t0+1/3*t2
a2.plot(x,t0,'o-',ms=4,label='петлі σ=0'); a2.plot(x,t2,'s-',ms=4,label='петлі σ=2'); a2.plot(x,tm,'k^-',ms=4,label='середнє')
c=tm[x==0.1][0]/0.01; a2.plot(al,c*al**2,':',color='gray',label='∝ α² (лінійний відгук)')
a2.axhline(0.0196,color='#c0392b',lw=0.8); a2.text(0.42,0.024,'ціна скручення у світі = виграш ΔV',fontsize=7,color='#c0392b')
a2.set_xlabel('α'); a2.set_ylabel('⟨1 − cos Φ⟩ на t 25'); a2.set_ylim(0,0.3); a2.legend(fontsize=7); a2.set_title('б) статичне скручення Матриці',fontsize=9)
fig.suptitle('(452) Фази петель Матриці 3D: рівняння Адлера, Кун L8',fontsize=10); fig.tight_layout()
fig.savefig('/mnt/project-files/figures/fig304_loop_locking.png'); print('ok')
