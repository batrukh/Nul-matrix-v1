"""(354) малюнок: посаджений світ у 3D (база 8.11) гасне за ≈1 од.; складники сили на ψ у кулі."""
import json, glob, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8)); out = {}
for d in sorted(glob.glob('/mnt/project-files/model/s52/r351/plant/s*_pr*')):
    j = json.load(open(glob.glob(d + '/bal_*.json')[0])); b = j['bal']; lab = d.split('/')[-1].replace('s', 's0 ', 1).replace('_pr', ', R=').replace('.', ',')
    t = [r['t'] for r in b]
    ax[0].plot(t, [r['psi'] for r in b], label=lab); ax[1].plot(t, [r['cost'] for r in b], label=lab); ax[2].plot(t, [r['qhol'] for r in b], label=lab)
    out[d.split('/')[-1]] = b[:7]
ax[0].axhline(0, c='k', lw=.8); ax[0].set(xlim=(0, 10), xlabel='t', ylabel='середнє ψ у посадженій кулі', title='(а) посаджений світ тане за ≈1 од. часу')
ax[1].set(xlim=(0, 5), xlabel='t', ylabel='сила ε0 на ψ (ціна нестрункості петель)', title='(б) переважає ціна входу в ψ>0')
ax[2].set(xlim=(0, 10), xlabel='t', ylabel='середня |фаза петлі| у кулі', title='(в) петлі під світом не вирівнюються')
for a in ax: a.legend(fontsize=7)
fig.suptitle('(354) 3D, тетраедри Куна, база 8.11: світ, посаджений у найбільшу пласку ділянку, гасне при s0 3,0 і 3,5, R 4 і 6', fontsize=11)
fig.tight_layout(); fig.savefig('/mnt/project-files/figures/fig251_3d_posadka.png', dpi=130); json.dump(out, open('ana354.json', 'w'))
