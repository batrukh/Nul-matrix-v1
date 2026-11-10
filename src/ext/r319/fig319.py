"""(319) малюнок: дисперсія вакууму без/з директором, маси при k=0, профілі світу-диска."""
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = '/mnt/project-files/model/s52/r319/'
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
cols = dict(base='#555555', **{'vac0.01': '#2a7fd4', 'vac0.03': '#d4572a'}); lab = dict(base='база (без директора)', **{'vac0.01': 'директор μ 0,01', 'vac0.03': 'директор μ 0,03'})
for t in ['base', 'vac0.01', 'vac0.03']:
    d = json.load(open(D + f'disp_{t}.json')); k = np.array(d['kx']); B = d['branches']
    mk = dict(base='s', **{'vac0.01': 'o', 'vac0.03': '^'})[t]; first = True
    for kk, b in zip(k, B):
        b = np.array(b); b = b[b > 1e-3] if t == 'base' else b      # у базі директор розчеплений (μ = k = 0) — нульові гілки прибрано
        b = b[b < 0.6]; ax[0].scatter(np.full(len(b), kk), b, s=14, marker=mk, facecolor='none', edgecolor=cols[t], label=lab[t] if first else None); first = False
ax[0].plot([0, 0.6], [0, 0.6], 'k:', lw=0.8, label='ω = k (c = 1)'); ax[0].set_ylim(0, 0.6); ax[0].set_xlabel('k'); ax[0].set_ylabel('ω'); ax[0].legend(fontsize=8); ax[0].set_title('А. Нижні гілки вакууму')
for i, t in enumerate(['base', 'vac0.01', 'vac0.03']):
    d = json.load(open(D + f'disp_{t}.json')); w = [q['w'] for q in d['k0'] if not (t == 'base' and q['n'] > 0.5)]
    w = np.array(w); w = w[(w > 1e-3) & (w < 1.2)]
    for x in np.unique(np.round(w, 3)):
        c = (np.round(w, 3) == x).sum(); ax[1].hlines(x, i - 0.3, i + 0.3, color=cols[t], lw=2); ax[1].text(i + 0.32, x, f'{x:.3f}×{c}', fontsize=7, va='center')
ax[1].set_xticks([0, 1, 2]); ax[1].set_xticklabels(['база', 'μ 0,01', 'μ 0,03']); ax[1].set_ylabel('маса ω(k=0)'); ax[1].set_title('А. Маси (безмасові не показано)')
ax[1].set_xlim(-0.5, 2.9)
try:
    for t, c in [('base', '#555555'), ('mu001', '#2a7fd4'), ('mu003', '#d4572a')]:
        d = json.load(open(D + f'disk_{t}.json')); r = [q['r'] for q in d['prof']]
        ax[2].plot(r, [q['t2'] for q in d['prof']], '-', color=c, label=f'trΦ² {t}')
        if t != 'base': ax[2].plot(r, [q['n_uv'] for q in d['prof']], '--', color=c, label=f'|n у площині фотона|² {t}')
    ax[2].axvline(d['R'], color='k', lw=0.6, ls=':'); ax[2].set_xlabel('r'); ax[2].legend(fontsize=7); ax[2].set_title('В. Світ-диск R 18 (ψ заморожене)')
except FileNotFoundError: pass
plt.tight_layout(); plt.savefig('/mnt/project-files/figures/fig214_director_base.png', dpi=130)
