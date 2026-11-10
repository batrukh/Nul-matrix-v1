"""(320) малюнок: спектр сектору q01=1 уздовж обміну, перекриття кроків, густина зв'язаних мод, фази обміну."""
import json, numpy as np, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '/mnt/project-files/model/s52/r320'); from pair import *
D = '/mnt/project-files/model/s52/r320'; L = 48
def tag(k): return f'L48_R8_A0.5_w2_N24_k{k//2}' if k % 2 == 0 else f'L48_R8_A0.5_w2_N48_k{k}'
fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
for k in range(49):
    w = np.sqrt(np.abs(np.load(f'{D}/modes/{tag(k)}.npz')['w2']))[:10]
    ax[0].scatter(np.full(len(w), k/48), w, s=8, c=['#d4572a' if x < 0.30 else '#888888' for x in w])
ax[0].axhline(0.331, color='k', ls=':', lw=0.8); ax[0].set_xlabel('φ / π'); ax[0].set_ylabel('ω'); ax[0].set_title('а) сектор заряду G01: зв\'язані (оранж.) і поріг')
b48 = json.load(open(f'{D}/braid_N48_k.json')); b24 = json.load(open(f'{D}/braid_N24_k.json'))
ax[1].plot(np.arange(1, 49)/48, b48['smins'], 'o-', ms=3, label='крок π/48'); ax[1].plot(np.arange(1, 25)/24, b24['smins'], 's-', ms=3, label='крок π/24')
ax[1].set_ylim(0, 1.02); ax[1].set_xlabel('φ / π'); ax[1].set_ylabel('мін. сингулярне перекриття'); ax[1].legend(fontsize=8); ax[1].set_title('б) перекриття сусідніх кроків')
z = np.load(f'{D}/modes/{tag(0)}.npz'); Va = z['Va'][..., :8]; rho = np.sum(Va**2, (2, 3)); Sx = pos(L)
ax[2].scatter(Sx[..., 0].ravel(), Sx[..., 1].ravel(), c=rho.ravel()/rho.max(), s=14, cmap='magma_r', vmin=0, vmax=1); v1, v2 = verts(L, 8, 0.0)
th = np.linspace(0, np.pi, 50); c = centre(L); ax[2].plot(c[0] + 8*np.cos(th), c[1] + 8*np.sin(th), 'c--', lw=1); ax[2].plot([v1[0], v2[0]], [v1[1], v2[1]], 'b-', lw=1)
ax[2].set_xlim(c[0] - 16, c[0] + 16); ax[2].set_ylim(c[1] - 12, c[1] + 12); ax[2].set_aspect('equal'); ax[2].set_title('в) густина 8 зв\'язаних мод (φ = 0), розріз Z, шлях')
eff = dict(two=[-0.9859, 0.0, 0.9859, 1.0], one=[-0.5115, -0.4974, 0.4974, 0.5115])
xs = {'ефективна (313)': (eff['one'], eff['two']), 'повна, π/24': (b24['phases1'], b24['two_one_per_vortex']), 'повна, π/48': (b48['phases1'], b48['two_one_per_vortex'])}
for i, (lab, (o, t)) in enumerate(xs.items()):
    ax[3].scatter(np.full(4, i) - 0.12, o, marker='o', c='#2a7fd4', label='1 квант' if i == 0 else None)
    ax[3].scatter(np.full(4, i) + 0.12, t, marker='^', c='#d4572a', label='2 кванти, по одному на вихорі' if i == 0 else None)
ax[3].scatter([1.3, 2.3], [b24['AA_phase'], b48['AA_phase']], marker='*', s=120, c='k', label='основний стан A_L A_R')
ax[3].axhline(1, color='k', lw=0.5, ls=':'); ax[3].axhline(-1, color='k', lw=0.5, ls=':'); ax[3].axhline(0, color='k', lw=0.5, ls=':')
ax[3].set_xticks(range(3)); ax[3].set_xticklabels(list(xs)); ax[3].set_ylabel('фаза обміну / π'); ax[3].legend(fontsize=7, loc='center left'); ax[3].set_title('г) фази обміну (±1 = ферміон)')
plt.tight_layout(); plt.savefig('/mnt/project-files/figures/fig215_full_exchange.png', dpi=120)
