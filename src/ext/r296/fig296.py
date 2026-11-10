"""(296) рис.: кутовий момент на квант для обертової хвилі фотона довкола вихору Аліси і без нього."""
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.special import jv
D = '/mnt/project-files/model/s52/r296/'
ld = lambda m, n: json.load(open(D + f'spin_{m}_nu{n:g}.json'))
CC, CV = '#2a6fdb', '#d9822b'
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
# (a) ν_eff проти заданого ν
for m, ns, c, mk, lab in [('ctrl', [2, 3, 4], CC, 'o', 'без вихору (цілі m)'), ('vort', [2.5, 3.5], CV, 's', 'довкола вихору (m + ½)')]:
    v = [ld(m, n)['ratio_wc'] for n in ns]; ax[0].plot(ns, v, mk, color=c, ms=10, label=lab)
    for n, y in zip(ns, v): ax[0].annotate(f'{y:.2f}', (n, y), xytext=(8, -14), textcoords='offset points', fontsize=10)
xx = np.array([1.7, 4.3]); ax[0].plot(xx, xx, color='0.6', lw=1, zorder=0)
for k in np.arange(2, 4.6, 0.5): ax[0].axhline(k, color='0.92', lw=0.8, zorder=-1)
ax[0].set_xlabel('кутове число хвилі ν (задане)'); ax[0].set_ylabel('L·ω / H  (кутовий момент на квант)')
ax[0].set_title('а) кутовий момент на квант = ν'); ax[0].legend(loc='upper left', frameon=False)
# (b) у часі, нормовано на контроль ν=3
rc = json.load(open(D + 'spin_ctrl_nu3.json'))['rec']; Rc = np.array([q['L']*q['w']/q['Hc'] for q in rc]); t = [q['t'] for q in rc]
for m, n, c, ls in [('ctrl', 2, CC, '-'), ('ctrl', 4, CC, '-'), ('vort', 2.5, CV, '--'), ('vort', 3.5, CV, '--')]:
    r = ld(m, n)['rec']; R = np.array([q['L']*q['w']/q['Hc'] for q in r]); ax[1].plot(t, 3*R/Rc, ls, color=c, lw=2)
    ax[1].text(t[-1] + 0.5, 3*R[-1]/Rc[-1], f'ν={n:g}', va='center', fontsize=10)
ax[1].set_xlim(0, 34); ax[1].set_xlabel('час t'); ax[1].set_ylabel('3·(L·ω/H) / (те саме для контролю ν=3)')
ax[1].set_title('б) у часі (спільні похибки методу зняті)'); ax[1].grid(color='0.92')
# (c) початкова хвиля B ∝ J_ν cos(νθ) довкола вихору, ν=2,5
x = np.linspace(-30, 30, 401); Xg, Yg = np.meshgrid(x, x); r = np.hypot(Xg, Yg)
th = np.mod(np.arctan2(Yg, Xg) - np.pi/2, 2*np.pi); B = np.exp(-(r/24)**4)*jv(2.5, 0.8*r)*np.cos(2.5*th)
ax[2].imshow(B, extent=[-30, 30, -30, 30], origin='lower', cmap='RdBu_r', vmin=-0.3, vmax=0.3)
ax[2].plot([0, 0], [0, 30], color='k', lw=2); ax[2].plot(0, 0, 'ko', ms=6)
ax[2].text(1.5, 24, 'розріз:\nфотон → −фотон', fontsize=10); ax[2].set_xticks([]); ax[2].set_yticks([])
ax[2].set_title('в) хвиля ν = 2,5: знак міняється через розріз')
fig.suptitle('(296) Вихор Аліси: квант фотона довкола вихору має півцілий кутовий момент (L192, база κa 1, κb 2)', fontsize=12)
fig.tight_layout(); fig.savefig('/mnt/project-files/figures/fig194_vortex_spin.png', dpi=130)
