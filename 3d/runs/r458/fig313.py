import json,glob,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
sets=[('runs/r5_*_s?_A.json','C3','r 5, затвор А'),('runs/r25_*_s?_A.json','C1','r 25, затвор А'),('runs/r100_*_s?.json','C2','r 100 (старий затвор)'),('runs/r5_*_s?.json','C7','r 5, старий затвор (світ теж відкриває ріст)')]
for j,(pat,c,lab) in enumerate(sets):
    ends=[]
    for i,f in enumerate(sorted(glob.glob(pat))):
        H=json.load(open(f))['hist']; t=[h['t'] for h in H]; y=[h['nlat']/4096 for h in H]; ends.append(y[-1])
        ax[0].plot(t,y,c=c,lw=1,alpha=.7,ls='--' if j==3 else '-',label=lab if i==0 else None)
    ax[1].scatter([j]*len(ends),ends,c=c)
ax[0].set_xlim(0,130); ax[0].set_xlabel('час Матриці'); ax[0].set_ylabel('частка ящика під ґраткою'); ax[0].legend(fontsize=8); ax[0].set_title('Ґратка з поля PFC під живою Матрицею')
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(['r 5\nА','r 25\nА','r 100\nстарий','r 5\nстарий'],fontsize=8)
ax[1].axhspan(0.60,0.85,color='gray',alpha=.15,label='передбачена зупинка r 5'); ax[1].axhline(0.95,ls=':',c='k')
ax[1].set_ylabel('частка ящика наприкінці'); ax[1].legend(fontsize=8); ax[1].set_title('Повільна ґратка: всесвіт = світ і стоїть')
plt.tight_layout(); plt.savefig('/mnt/project-files/figures/fig313_stageB_gateA.png',dpi=130)
