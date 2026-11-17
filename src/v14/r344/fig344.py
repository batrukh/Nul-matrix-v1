import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = json.load(open('lens344.json'))
fig, ax = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw=dict(width_ratios=[1, 1, 0.8]))
for a, n, tt in ((ax[0], 6, 'а) пласка ґратка: фронт проходить рівно'), (ax[1], 5, 'б) конус (вузол з 5 сусідами, виріз 60°):\nверхня й нижня половини фронту позаду сходяться')):
    d = np.load(f'lens_n{n}.npz'); P, u = d['P'], d['s80']; m = d['ring'] < 118
    a.scatter(P[m, 0], P[m, 1], c=u[m], s=1.2, cmap='RdBu_r', vmin=-0.6, vmax=0.6, rasterized=True)
    a.plot([0], [0], 'k*', ms=10)
    if n == 5:
        for s in (1, -1): a.plot([0, 118*np.cos(np.radians(30))], [0, s*118*np.sin(np.radians(30))], 'k--', lw=0.8)
        a.text(60, 0, 'виріз\n(краї склеєні)', ha='center', va='center', fontsize=8)
    a.set_aspect('equal'); a.set_xlim(-120, 120); a.set_ylim(-120, 120); a.set_title(tt, fontsize=10); a.set_xticks([]); a.set_yticks([])
a = ax[2]
rel = {6: R['6']['dir_up'] - R['6']['dir_dn'], 5: 60 + (R['5']['dir_up'] - R['5']['dir_dn'])}
a.bar([0, 1, 2], [abs(rel[6]), rel[5], 60], color=['#999999', '#d9480f', '#2a6fdb'])
for i, v in enumerate([abs(rel[6]), rel[5], 60]): a.text(i, v + 1.5, f'{v:.1f}°', ha='center')
a.set_xticks([0, 1, 2]); a.set_xticklabels(['пласка', 'конус\n(модель)', '2D-ЗТВ:\nмаса з\n8πGm = 60°'], fontsize=9); a.set_ylim(0, 75)
a.set_ylabel('кут між двома фронтами позаду, °')
a.set_title('в) лінза: два зображення під кутом,\nрівним вирізу (як маса в 2D-ЗТВ)', fontsize=10)
plt.tight_layout(); plt.savefig('/mnt/project-files/figures/fig240_disclination_lens.png', dpi=120)
print(rel)
