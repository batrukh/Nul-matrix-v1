import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
H=['0,30\n(8)','0,33\n(8)','0,35\n(8)','0,36\n(16)','0,37\n(8)','0,39\n(8)']; n=np.array([8,8,8,16,8,8])
F=np.array([1,1,1,5,1,8]); I=np.array([0,0,2,4,3,0]); G=np.array([0,0,1,1,1,0]); Z=n-F-I-G
fig,ax=plt.subplots(figsize=(8,4.4)); x=np.arange(6); b=np.zeros(6)
for v,c,l in ((Z,'0.65','нічого'),(I,'C2','сталий острів'),(G,'C1','острів росте/міняється >20 %'),(F,'C0','світ на всю ґратку')):
    ax.bar(x,100*v/n,bottom=b,color=c,label=l); b+=100*v/n
for i in range(6): ax.text(x[i],100*(Z[i]+I[i]/2)/n[i],f'{I[i]}/{n[i]}' if I[i] else '',ha='center',va='center',fontsize=9,color='w')
ax.set_xticks(x,H); ax.set_xlabel('нахил h (у дужках — зерен); α 0,25, s0 3,0'); ax.set_ylabel('% зерен'); ax.set_ylim(0,100)
ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(0.5,-0.3),ncol=4)
ax.set_title('(425В) Острови — смуга h 0,35–0,37 перед різким переходом до світу на всю ґратку\nКун L16, T600; 0,30/0,36/0,39 — раніше, 0,33/0,35/0,37 — нічний скан',fontsize=10); fig.tight_layout()
fig.savefig('/mnt/project-files/figures/fig298_3d_smuga_ostroviv.png',dpi=130); fig.savefig('/tmp/claude-0/-home-user/8a03f883-4d78-5ab1-a5f0-9aa077653cc0/scratchpad/fig298.png',dpi=130)
