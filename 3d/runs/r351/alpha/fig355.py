import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['axes.formatter.use_locale'] = False
d = json.load(open('/mnt/project-files/model/s52/r351/alpha/ana355.json')); A = [0.3, 0.4, 0.5, 0.7, 1.0]
uk = lambda x: f'{x:g}'.replace('.', ',')
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for s0, mk in ((3.0, 'o'), (3.5, 's')):
    r = [d[f'{s0}_{a}'] for a in A]
    ax[0].plot(A, [x['psimax'] for x in r], mk+'-', label=f'запас {uk(s0)}')
    ax[1].plot(A, [x['flat'] for x in r], mk+'-', label=f'пласкі петлі, запас {uk(s0)}'); ax[1].plot(A, [x['boil'] for x in r], mk+'--', alpha=.6, label=f'кипіння, запас {uk(s0)}')
ax[0].axhline(0, c='k', lw=.8)
ax[0].set(xlabel='обертання α', ylabel='найбільше ψ при t=100', title='(а) ψ>0 ніде: 0 світів у 8 прогонах'); ax[0].legend(fontsize=8)
ax[1].set(xlabel='обертання α', ylabel='частка (t≥50)', title='(б) менше α: петлі пласкі, кипіння мало'); ax[1].legend(fontsize=7)
for a in A:
    x = d[f'3.0_{a}']; ax[2].plot(x['tser'], x['flat_t'], label=f'α {uk(a)}')
ax[2].set(xlabel='t', ylabel='частка пласких петель', title='(в) пласка «Матриця» без світу (запас 3,0)'); ax[2].legend(fontsize=7)
for a in ax[:2]: a.set_xticks(A); a.set_xticklabels([uk(v) for v in A])
fig.suptitle('(355) 3D, тетраедри Куна, база 8.11, L16, T100: прохід обертання α. Петлі вирівнюються, але світи не народжуються (α 1 — з (351))', fontsize=10)
fig.tight_layout(); fig.savefig('/mnt/project-files/figures/fig253_3d_obertannia.png', dpi=130)
