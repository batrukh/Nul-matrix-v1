import ast,json,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
out={}; fig,ax=plt.subplots(1,3,figsize=(13,3.8))
for s in range(1,9):
    recs=[ast.literal_eval(l) for l in open(f'runs/M5_seed{s}/log') if l.startswith('{')]
    d=np.load(f'runs/M5_seed{s}/prof420.npz'); t=d['t']
    out[s]=dict(maxbig=max(r['big'] for r in recs), z_core_end=float(d['z'][-1][:2].mean()), psi_core_end=float(d['psi'][-1][:2].mean()),
                z_chaos_end=float(d['z'][-1][8:].mean()), boil_t10=recs[9].get('boil'), boil_end=recs[-1].get('boil'))
    ax[2].plot(t,d['z'][:,:2].mean(1),label=f'зерно {s}')
    if s==1:
        for i,c in zip((0,9,19,49,299),plt.cm.viridis(np.linspace(0,.9,5))):
            r=np.arange(d['psi'].shape[1]); ax[0].plot(r,d['psi'][i],color=c,label=f't{int(t[i])}'); ax[1].plot(r,d['z'][i],color=c,label=f't{int(t[i])}')
for a in ax[:2]: a.axvline(5,ls='--',c='k',lw=.8); a.set_xlabel('відстань від центру латки r'); a.set_xlim(0,10)
ax[0].set_ylabel('середнє ψ'); ax[0].axhline(0,c='gray',lw=.6); ax[0].set_title('(а) ψ за радіусом, зерно 1'); ax[0].legend(fontsize=7)
ax[1].set_ylabel('середнє |z|² ниток'); ax[1].set_title('(б) запас ниток за радіусом, зерно 1')
ax[2].set_xlabel('час t'); ax[2].set_ylabel('|z|² у ядрі латки (r<2)'); ax[2].set_title('(в) ядро латки, 8 зерен'); ax[2].legend(fontsize=6,ncol=2)
plt.tight_layout(); plt.savefig('ana/fig288.png',dpi=130)
json.dump(out,open('ana/ana420.json','w'),indent=1); print(json.dumps(out))
