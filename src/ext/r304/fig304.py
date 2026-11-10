import numpy as np, json, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '.'); sys.path.insert(0, '/mnt/project-files/model/s52/r268/code'); import hc, phi90 as PH
L = 96; XA, YA = hc.coordsAB(L)
R = {k: json.load(open(f'r304_{k}.json')) for k in ['vort_g1.0', 'core_g1.0', 'vort_g0.5']}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
lab = {'vort_g1.0': 'вихор бази, g=1', 'core_g1.0': 'те саме ядро без скруту, g=1', 'vort_g0.5': 'вихор бази, g=0,5'}
for i, (k, r) in enumerate(R.items()):
    E = np.array(r['E']); ax[0].hlines(E/r['gap'], i - 0.3, i + 0.3, color='C3' if k != 'core_g1.0' else 'C0', lw=1.5)
    z0 = np.abs(E) < 0.01*r['gap']
    if z0.any(): ax[0].plot([i]*z0.sum(), E[z0]/r['gap'], 's', color='C3', ms=9); ax[0].text(i + 0.33, 0.04, f'{z0.sum()} нул.', fontsize=9, color='C3')
ax[0].axhspan(-1, 1, color='0.92', zorder=-1); ax[0].axhline(0, color='k', lw=0.3, ls=':')
ax[0].set_xticks(range(3)); ax[0].set_xticklabels([lab[k] for k in R], fontsize=8); ax[0].set_ylim(-1.15, 1.15)
ax[0].set_ylabel('E / щілина'); ax[0].set_title('а) рівні біля E=0 (сіре — щілина)')
d = np.load('r304_vort_g1.0_dens.npz'); sel = np.abs(d['E']) < 0.01; rho = d['dens'][:, sel].sum(1)
cen = d['cen']
sc = ax[1].scatter(XA, YA, c=rho, s=1.2, cmap='magma', vmax=rho.max()*0.6); ax[1].set_aspect('equal')
ax[1].set_title(f'б) густина {sel.sum()} нульових мод (g=1, L96)'); plt.colorbar(sc, ax=ax[1], shrink=0.8)
for (cx, cy, _) in cen: ax[1].plot(cx, cy, 'c+', ms=10)
for k, c in [('vort_g1.0', 'C3'), ('vort_g0.5', 'C1')]:
    d = np.load(f'r304_{k}_dens.npz'); sel = np.abs(d['E']) < 0.01; rho = d['dens'][:, sel].sum(1)
    r1 = PH.tdist(XA, YA, d['cen'][0][0], d['cen'][0][1], L); bins = np.arange(0, 20, 1.0)
    h = np.array([rho[(r1 >= b) & (r1 < b + 1)].mean() for b in bins]); ax[2].semilogy(bins + 0.5, h, 'o-', color=c, label=lab[k])
ax[2].set_xlabel('відстань від ядра v1'); ax[2].set_ylabel('густина на трикутник'); ax[2].legend(fontsize=8); ax[2].set_title('в) профіль нульових мод')
fig.suptitle('(304) Ферміон (вектор SO(5)) на соті поверх вихору Аліси бази: по 2 нульові моди на вихор', fontsize=11)
plt.tight_layout(); plt.savefig('fig200_honeycomb_zero_modes.png', dpi=130)
