"""(320) paper figure (English): bound-mode density at φ=0, overlaps, exchange phases. Spectrum along the path: numbers in text (modes/ for k>0 deleted 10.11)."""
import json, numpy as np, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '/mnt/project-files/model/s52/r320'); from pair import *
D = '/mnt/project-files/model/s52/r320'; L = 48
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
z = np.load(f'{D}/modes/L48_R8_A0.5_w2_N24_k0.npz'); Va = z['Va'][..., :8]; rho = np.sum(Va**2, (2, 3)); Sx = pos(L)
ax[0].scatter(Sx[..., 0].ravel(), Sx[..., 1].ravel(), c=rho.ravel()/rho.max(), s=14, cmap='magma_r', vmin=0, vmax=1); v1, v2 = verts(L, 8, 0.0)
th = np.linspace(0, np.pi, 50); c = centre(L); ax[0].plot(c[0] + 8*np.cos(th), c[1] + 8*np.sin(th), 'c--', lw=1); ax[0].plot([v1[0], v2[0]], [v1[1], v2[1]], 'b-', lw=1)
ax[0].set_xlim(c[0] - 16, c[0] + 16); ax[0].set_ylim(c[1] - 12, c[1] + 12); ax[0].set_aspect('equal'); ax[0].set_title('(a) bound odd modes at φ = 0')
b48 = json.load(open(f'{D}/braid_N48_k.json')); b24 = json.load(open(f'{D}/braid_N24_k.json'))
ax[1].plot(np.arange(1, 49)/48, b48['smins'], 'o-', ms=3, label='step π/48'); ax[1].plot(np.arange(1, 25)/24, b24['smins'], 's-', ms=3, label='step π/24')
ax[1].set_ylim(0, 1.02); ax[1].set_xlabel('φ / π'); ax[1].set_ylabel('min. overlap of neighbouring steps'); ax[1].legend(fontsize=8); ax[1].set_title('(b) adiabatic transport')
eff = dict(two=[-0.9859, 0.0, 0.9859, 1.0], one=[-0.5115, -0.4974, 0.4974, 0.5115])
xs = {'effective': (eff['one'], eff['two']), 'full, π/24': (b24['phases1'], b24['two_one_per_vortex']), 'full, π/48': (b48['phases1'], b48['two_one_per_vortex'])}
for i, (lab, (o, t)) in enumerate(xs.items()):
    ax[2].scatter(np.full(4, i) - 0.12, o, marker='o', c='#2a7fd4', label='one quantum' if i == 0 else None)
    ax[2].scatter(np.full(4, i) + 0.12, t, marker='^', c='#d4572a', label='two quanta, one per vortex' if i == 0 else None)
ax[2].scatter([1.3, 2.3], [b24['AA_phase'], b48['AA_phase']], marker='*', s=120, c='k', label='ground state $A_LA_R$')
for y in (-1, -.5, 0, .5, 1): ax[2].axhline(y, color='grey', lw=.4, ls=':')
ax[2].set_xticks(range(3)); ax[2].set_xticklabels(list(xs)); ax[2].set_ylabel('exchange phase / π'); ax[2].legend(fontsize=7, loc='center left'); ax[2].set_title('(c) exchange phases (±1: fermion)')
plt.tight_layout(); plt.savefig('/mnt/project-files/paper/arxiv-v1/figs/fig11_exchange.pdf')
