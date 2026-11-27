import ast,os,glob,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R='/mnt/project-files/model/s52/r425/runs'; N=4096
def load(d):
    D=[ast.literal_eval(l) for l in open(d+'/log') if l.startswith('{')]; return np.array([x['t'] for x in D]),np.array([x['big'] for x in D])
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
isl={6:'з6',7:'з7',11:'з11',12:'з12',5:'з5 (росте)'}
for s in range(1,17):
    t,b=load(f'{R}/M_seed{s}')
    if s in isl: ax[0].plot(t,100*b/N,lw=2,label=isl[s])
    else: ax[0].plot(t,100*b/N,color='0.75',lw=0.8)
ax[0].axvspan(300,600,color='C2',alpha=0.08); ax[0].set_xlabel('час t'); ax[0].set_ylabel('найбільший світ, % ґратки')
ax[0].set_title('h 0,36: 16 зерен (сірі — світ на всю ґратку або нічого)'); ax[0].legend(fontsize=8); ax[0].set_ylim(-2,100)
H=['0,30 (раніше, 8 зерен)','0,36 (425)','0,39 (425)']; nothing=[7/8*100,6/16*100,0]; island=[0,4/16*100,0]; full=[1/8*100,5/16*100,100]; grow=[0,1/16*100,0]
x=np.arange(3); ax[1].bar(x,nothing,color='0.6',label='нічого'); ax[1].bar(x,island,bottom=nothing,color='C2',label='сталий острів')
b2=np.add(nothing,island); ax[1].bar(x,grow,bottom=b2,color='C1',label='острів росте'); ax[1].bar(x,full,bottom=b2+np.array(grow),color='C0',label='світ на всю ґратку')
ax[1].set_xticks(x,H); ax[1].set_xlabel('h (α 0,25)'); ax[1].set_ylabel('% зерен'); ax[1].set_title('Острови — лише біля переходу'); ax[1].legend(fontsize=8,loc='upper center',bbox_to_anchor=(0.5,-0.18),ncol=4)
fig.suptitle('(425) Сталі острови в живій Матриці, Кун L16, s0 3,0, α 0,25, T600',fontsize=11); fig.tight_layout()
fig.savefig('/mnt/project-files/figures/fig295_3d_ostrovy_T600.png',dpi=130); fig.savefig('/tmp/claude-0/-home-user/8a03f883-4d78-5ab1-a5f0-9aa077653cc0/scratchpad/fig295.png',dpi=130)
