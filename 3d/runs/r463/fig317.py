import ast,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
N=8**4; arms=[("N0","на нитку, h=0","tab:blue"),("N15","на нитку, h=0,15","tab:red"),("C15","чинна модель, h=0,15","tab:gray")]
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
for a,lab,c in arms:
    for s in range(1,9):
        R=[]
        for l in open(f"runs/{a}_seed{s}/log"):
            if l.startswith("{'A2'"):
                try: R.append(ast.literal_eval(l))
                except: pass
        t=[r['t'] for r in R]
        ax[0].plot(t,[r['nB']/N for r in R],c=c,lw=1.2,alpha=.8,label=lab if s==1 else None)
        ax[1].plot(t,[r['sd'] for r in R],c=c,lw=1,alpha=.7,label=lab if s==1 else None)
ax[0].axvline(50,ls=':',c='k'); ax[0].axhline(.5,ls=':',c='k')
ax[0].set_xlabel('час t'); ax[0].set_ylabel('частка вузлів з ψ>0 (світ)'); ax[0].set_title('а) Світ у 4D (Кун L8, по 8 зерен)')
ax[1].set_xlabel('час t'); ax[1].set_ylabel('розкид |z|² (sd)'); ax[1].set_title('б) Кипіння Матриці')
ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
fig.suptitle('(463) Зв\'язок на нитку в 4D: Матриця не розпадається; світ пізно — у 3/8 зерен лише при h=0,15',fontsize=10)
fig.tight_layout(); fig.savefig('/mnt/project-files/figures/fig317_per_thread_4d.png',dpi=130)
