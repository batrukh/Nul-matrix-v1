import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
d = json.load(open('/mnt/project-files/model/s52/r351/ana351.json')); S = sorted({v['s0'] for v in d.values()})
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for seed, mk in ((1, 'o'), (2, 's')):
    r = [d[f'{s}_{seed}'] for s in S]
    ax[0].plot(S, [x['psimax'] for x in r], mk+'-', label=f'реалізація {seed}')
    ax[1].plot(S, [x['proto'] for x in r], mk+'-', label=f'протосвіти, реал. {seed}'); ax[1].plot(S, [x['boil'] for x in r], mk+'--', alpha=.6, label=f'кипіння, реал. {seed}')
ax[0].axhline(0, c='k', lw=.8)
ax[0].set(xlabel='запас s0', ylabel='найбільше ψ при t=100', title='(а) жоден вузол не має ψ>0 (0 світів у 12 прогонах)'); ax[0].legend(fontsize=8)
ax[1].set(xlabel='запас s0', ylabel='частка (t≥50)', title='(б) більше запасу: більше кипіння й протосвітів'); ax[1].legend(fontsize=7)
for s in S:
    x = d[f'{s}_1']; ax[2].plot(x['tser'], x['proto_t'], label=f's0 {s:g}'.replace('.', ','))
ax[2].set(xlabel='t', ylabel='частка протосвітів', title='(в) протосвіти в часі не ростуть (реалізація 1)'); ax[2].legend(fontsize=7)
fig.suptitle('(351) 3D, тетраедри Куна, база 8.11 (β 1,3, випадкові стріли), L16, T100: світи не народились', fontsize=11)
fig.tight_layout(); fig.savefig('/mnt/project-files/figures/fig250_3d_krok0.png', dpi=130)
