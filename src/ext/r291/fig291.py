import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2)); dE = 0.921; E0 = 0.9627
for f, c, lab in [('move_g0.004_gam0.json', 'C0', 'g = 0,004, без тертя'), ('move_g0.008_gam0.json', 'C3', 'g = 0,008, без тертя'), ('move_g0.008_gam0.2.json', '0.4', 'g = 0,008, тертя бази 0,2')]:
    d = json.load(open(f)); t = np.array([r['t'] for r in d['rec']]); x = np.array([r['p'][0][0] for r in d['rec']]); x -= x[0]
    ax[0].plot(t, x, 'o', ms=3, color=c, label=lab)
    if d['gam'] == 0: ax[0].plot(t, 0.5*dE*d['g']/E0*t**2, '--', color=c, lw=0.8)
ax[0].plot([], [], 'k--', lw=0.8, label='передбачення, якби m = E₀ (c = 1)')
ax[0].set_xlabel('час'); ax[0].set_ylabel('зсув вихору вздовж сили'); ax[0].legend(fontsize=8); ax[0].set_title('Вихор у нахилі потенціалу рівноприскорений, без застрягання', fontsize=9)
lab = ['E₀ (енергія ядра)', 'm_I = F/прискорення\n(g 0,004 / 0,008)', 'm опору (тертя 0,2)', 'E₀/c² для c = 0,8\n(частини з ланками)']
val = [E0, dE/0.68, dE*0.008/(0.2*0.0275), 1.41]
ax[1].bar(range(4), val, color=['0.6', 'C3', '0.4', 'C2']); ax[1].set_xticks(range(4)); ax[1].set_xticklabels(lab, fontsize=8)
for i, v in enumerate(val): ax[1].text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
ax[1].set_ylabel('маса'); ax[1].set_title('Інерція ≈ 1,4·E₀: «світло» для вихору ≈ 0,84, а не 1 (гіпотеза)', fontsize=9)
fig.suptitle('(291) Вихор Аліси як частинка (база b = 0,1, синтетичний світ L96)', fontsize=10)
fig.tight_layout(); fig.savefig('/mnt/project-files/figures/fig189_vortex_inertia.png', dpi=130)
