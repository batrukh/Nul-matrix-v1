import json, glob, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
f, ax = plt.subplots(1, 4, figsize=(17, 4.2))
for d, c in [('s1_d1', 'C0'), ('s2_d1', 'C1')]:
    dt = json.load(open(glob.glob(f'../r321/{d}/*_dtrack.json')[0])); t = [q['k']*0.05 for q in dt]
    ax[0].plot(t, [max(q['nd'], .5) for q in dt], c, lw=2, label=f'зерно {d[1]}: дисклінації n')
    ax[0].plot(t, [max(q['nz2'], .5) for q in dt], c, ls=':', label=f'зерно {d[1]}: вихор Аліси')
for i, s in enumerate(['s3', 's4', 's5', 's6']):
    dt = json.load(open(glob.glob(f'{s}/*_dtrack.json')[0])); ax[0].plot([q['k']*0.05 for q in dt], [max(q['nd'], .5) for q in dt], color=f'C{i+2}', lw=1, alpha=.7, label=f'зерно {s[1]} (T 300)')
ax[0].set(yscale='log', xlabel='час t', ylabel='трикутники-дефекти у світах', title='а) дисклінації замерзають до t 800'); ax[0].legend(fontsize=7)
lab = ['1 без n', '2 без n', '1 з n', '2 з n', '3 з n', '4 з n', '5 з n', '6 з n']; L = [90, 105, 30, 789, 37, 73, 31, 31]
ax[1].bar(range(8), L, color=['grey']*2 + ['C0']*6); ax[1].set_xticks(range(8)); ax[1].set_xticklabels(lab, rotation=45, fontsize=8)
ax[1].set(ylabel='найдовше життя вихору Аліси', title='б) вихор + дисклінація: 789 у зерні 2'); ax[1].axhline(105, color='grey', ls='--')
o = json.load(open('ana322b.json'))['out']
ax[2].bar([0, 1], [1000*o['seam'][0]/o['seam'][1], 1000*o['front'][0]/o['front'][1]], color=['C3', 'C7'])
for i, k in enumerate(['seam', 'front']): ax[2].text(i, 1000*o[k][0]/o[k][1] + 2, f"{o[k][0]} нових\nживі через 50: {100*o[k][3]/o[k][0]:.0f} %", ha='center', fontsize=8)
ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(['шов злиття світів', 'звичайний фронт росту'])
ax[2].set(ylabel='нових дисклінацій на 1000 нових вузлів світу', title='в) ідея 2: шрами від злиття', ylim=(0, 115))
R = [('1', 25, 8), ('2', 20, 7), ('3', 29, 8), ('4', 13, 2), ('5', 21, 6), ('6', 13, 3)]
ax[3].plot([r[1] for r in R], [r[2] for r in R], 'o', ms=8)
for r in R: ax[3].annotate(r[0], (r[1] + .4, r[2] + .15))
ax[3].set(xlabel='час, коли світи вкрили половину кінцевої площі', ylabel='дисклінацій у світах при t 150', title='г) ідея 1: повільніше — більше реліктів\n(Спірмен 0,91, p 0,01, 6 зерен)')
plt.tight_layout(); plt.savefig('/mnt/project-files/figures/fig217_relics_scars.png', dpi=110)
