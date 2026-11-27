"""(375C) Рис. 269: світ у 3D до T600 (α0,1, L16, 8 зерен)."""
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R='/mnt/project-files/model/s52/r375'; out={}
fig,ax=plt.subplots(1,2,figsize=(10,4))
for s in range(1,9):
    w=json.load(open(f'{R}/a0.1_seed{s}_T600/t3_s3_L16_s{s}_ell4.json'))['w2']
    t=[r['t'] for r in w]; b=[r['big']/4096 for r in w]
    d={round(r['t']):r['big'] for r in w}
    out[s]=dict(b300=d[300],b600=d[600],shrink=(d[300]-d[600])/d[300])
    ax[0].plot(t,b,lw=.8,label=f'зерно {s}')
ax[0].axvline(300,color='k',ls=':',lw=.8); ax[0].set_xlabel('час, од.'); ax[0].set_ylabel('найбільший світ, частка ґратки')
ax[0].set_title('(а) світ тримається до t600'); ax[0].legend(fontsize=7,ncol=2)
sh=[100*out[s]['shrink'] for s in out]
ax[1].bar(range(1,9),sh,color='#2ca02c'); ax[1].axhline(5,color='r',ls='--',lw=1,label='поріг критерію 5 %')
ax[1].set_ylim(0,6); ax[1].set_xlabel('зерно'); ax[1].set_ylabel('стиснення t300→t600, %'); ax[1].set_title('(б) стиснення ≤0,17 %, медіана 0'); ax[1].legend(fontsize=8)
fig.suptitle('(375C) 3D, Кун L16, α 0,1, s0 3,0: світ стійкий до T600 у 8/8 зерен (але один світ ≈87 % ґратки)',fontsize=10)
fig.tight_layout(); fig.savefig('/mnt/project-files/figures/fig269_3d_T600.png',dpi=130)
json.dump(out,open('anaC.json','w'),indent=1)
