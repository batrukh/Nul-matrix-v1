import json,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
d=json.load(open('ana452v.json'))['rows']; A=sorted(d,key=float)
x=[float(a) for a in A]; y=[d[a]['f2'] for a in A]; e=[d[a]['se'] for a in A]
x8=[float(a) for a in A if d[a]['f2L8'] is not None]; y8=[d[a]['f2L8'] for a in A if d[a]['f2L8'] is not None]
fig,ax=plt.subplots(1,2,figsize=(9,3.6))
ax[0].errorbar(x,y,e,fmt='o-',label='L16, 8 зерен'); ax[0].plot(x8,y8,'s--',color='gray',label='L8, 2 зерна (452)')
ax[0].axhline(2/3,color='r',ls=':',label='крок середнього поля (σ=0 лишаються)')
ax[0].axvspan(0.05,0.2,color='g',alpha=0.1,label='вікно світів (375), h 0,30')
ax[0].set_xlabel('α'); ax[0].set_ylabel('частка замкнених петель σ=2'); ax[0].legend(fontsize=7); ax[0].set_ylim(0.6,1.02)
ax[1].bar(x,[d[a]['nworld'] for a in A],width=0.03); ax[1].set_xlabel('α'); ax[1].set_ylabel('зерен зі світом до t25 (з 8)')
fig.suptitle('(452В) Замикання петель Матриці, Кун L16, h 0,36'); fig.tight_layout()
fig.savefig('/mnt/project-files/figures/fig452v.png',dpi=130)
