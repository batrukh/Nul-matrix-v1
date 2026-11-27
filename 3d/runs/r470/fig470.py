import json,glob,numpy as np,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(10,4),dpi=150)
arms=[(2.9,0.2,'C0'),(3.0,0.2,'C1'),(3.15,0.2,'C2'),(3.0,-0.2,'C3')]
for s0,mu,c in arms:
    fs=sorted(glob.glob(f"runs/s{s0:g}_*mu{mu:g}_*.json")) or [f for f in glob.glob("runs/*.json") if json.load(open(f))["s0"]==s0 and json.load(open(f))["mu"]==mu]
    Y=[];
    for f in fs:
        d=json.load(open(f));P=d["prof_rho"];r=np.array([float(k) for k in P]);y=np.array([v[0] for v in P.values()]);Y.append(y-y[(r<=8)][-1])
    y=np.mean(Y,0);m=(r>=1)&(r<=8)
    lab=f"s0 {s0:g}, μ {mu:+g}"
    ax[0].plot(r[m],y[m],'o-',color=c,ms=3,label=lab)
    ax[1].semilogy(r[m],np.abs(y[m])*r[m],'o-',color=c,ms=3,label=lab)
ax[0].axhline(0,color='k',lw=.5);ax[0].set_xlabel('відстань від джерела r (кроки ґратки)');ax[0].set_ylabel('δρ(r) (запас мінус фон)');ax[0].set_title('Ореол запасу довкола пробної маси');ax[0].legend(fontsize=8)
ax[1].set_xlabel('r');ax[1].set_ylabel('|δρ|·r (лог.)');ax[1].set_title('Нахил = −1/ξ: ближче до порога — далі');ax[1].legend(fontsize=8)
plt.tight_layout();plt.savefig('/mnt/project-files/figures/fig470.png')
