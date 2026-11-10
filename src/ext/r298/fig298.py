import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, sys, os
sys.path.insert(0, '.'); sys.path.insert(0, '..')
from cgauge import CGauge
plt.rcParams.update({'font.size': 10})
E0 = -917.015
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
runs = [('pair_D24s', '±1, d = 24', 'C0'), ('pair_D10s', '±1, d = 10', 'C2'), ('pair_D6s', '±1, d = 6', 'C3'), ('pair_D4s', '±1, d = 4', 'C1'),
        ('quad_D4s', '+1+1 (d = 4) і −1−1', 'C4'), ('ctrl_D24s', 'контроль: ядра без обертів', 'k'), ('pair_D24', '±1, потік в одному трикутнику', 'C7')]
for tag, lab, c in runs:
    d = json.load(open(f'vort_{tag}.json')); ax[0].plot(d['t'], np.array(d['E']) - E0, c=c, label=lab, ls='--' if tag == 'pair_D24' else '-')
ax[0].axvline(100, c='gray', lw=0.6, ls=':'); ax[0].text(102, 13, 'без тертя →', fontsize=8, color='gray')
ax[0].set_xlabel('час t'); ax[0].set_ylabel('енергія над вакуумом'); ax[0].set_ylim(-0.5, 16); ax[0].legend(fontsize=7.5, loc='center right')
ax[0].set_title('а) енергія: ≈3,5 на вихор, тримається')
f = np.load('vort_pair_D24s_fin.npz'); P, A = f['P'], f['A']; L = P.shape[0]
i, j = np.meshgrid(np.arange(L), np.arange(L), indexing='ij'); X = i + 0.5*j; Y = np.sqrt(3)/2*j
mag = np.sum(np.abs(P)**2, axis=(-2, -1))
g = CGauge(np.full((L, L), -1.), np.ones((3, L, L)), np.ones((2, L, L))); Fu, Fd = g.flux(A); w = lambda x: (x + np.pi) % (2*np.pi) - np.pi
im = ax[1].scatter(X.ravel() % L, Y.ravel(), c=mag.ravel(), s=4, cmap='viridis'); plt.colorbar(im, ax=ax[1], label='|Φ★|²')
ax[1].set_aspect('equal'); ax[1].set_title('б) |Φ★|² при t = 200: два ядра (n = +1, −1)'); ax[1].set_xticks([]); ax[1].set_yticks([])
im2 = ax[2].scatter(X.ravel() % L, Y.ravel(), c=(w(Fu) + w(Fd)).ravel(), s=4, cmap='RdBu', vmin=-0.15, vmax=0.15); plt.colorbar(im2, ax=ax[2], label='потік B у трикутниках')
ax[2].set_aspect('equal'); ax[2].set_title('в) потік нової U(1): ∓1 квант на вихор'); ax[2].set_xticks([]); ax[2].set_yticks([])
plt.tight_layout(); plt.savefig('/mnt/project-files/figures/fig196_integer_vortex.png', dpi=130); print('ok')
