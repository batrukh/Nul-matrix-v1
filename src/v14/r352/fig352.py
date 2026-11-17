import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, sys
sys.path.insert(0, '/mnt/project-files/model/s52/r348/code'); import simNlib as N
M = np.load('/mnt/project-files/model/s52/r352/maps_s1.npz'); A = json.load(open('/mnt/project-files/model/s52/r352/ana352.json'))
f = N.Fabric(96); L = 96; n = f.n
def lap(x): a, b = f.tail, f.head; d = x[b] - x[a]; return np.bincount(a, d, n) - np.bincount(b, d, n)
def block(x, k): return x.reshape(L//k, k, L//k, k).mean((1, 3)).ravel()
fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
v = np.percentile(np.abs(M['dl']), 98)
ax[0].imshow(M['dl'].reshape(L, L).T, cmap='RdBu_r', vmin=-v, vmax=v, origin='lower'); ax[0].contour((M['psi'] > 0).reshape(L, L).T, [0.5], colors='k', linewidths=0.6)
ax[0].set_title('(а) дефіцит кута δ (p 0,1), чорне — межі світів', fontsize=9)
ax[1].imshow(M['dE'].reshape(L, L).T, cmap='RdBu_r', origin='lower'); ax[1].contour((M['psi'] > 0).reshape(L, L).T, [0.5], colors='k', linewidths=0.6); ax[1].set_title('(б) енергія вузла E − Ē', fontsize=9)
ax[2].plot(block(M['dE'], 4), block(M['dl'], 4), '.', ms=3, alpha=0.6); ax[2].set_xlabel('E − Ē (блоки 4×4)'); ax[2].set_ylabel('δ (блоки 4×4)')
ax[2].set_title(f"(в) δ проти енергії: r = {np.corrcoef(block(M['dE'],4), block(M['dl'],4))[0,1]:.2f}", fontsize=9)
ax[3].plot(lap(M['A2n']), M['dl'], '.', ms=1, alpha=0.3, color='C3'); ax[3].set_xlabel('Δ|z|² (лапласіан сили ниток)'); ax[3].set_ylabel('δ')
ax[3].set_title(f"(г) δ проти Δ|z|²: r = {np.corrcoef(lap(M['A2n']), M['dl'])[0,1]:.2f}", fontsize=9)
for a in ax[:2]: a.set_xticks([]); a.set_yticks([])
rows = [(k, d['r_avg'], d['r_lapA2']) for k, d in A.items() if k.endswith('p0.1')]
plt.suptitle('(352) довжина нитки з її сили l = (w₊²/|z|²)^p: кривина є, але сидить на межах світів і йде за Δ|z|², не за енергією.  ' + '; '.join(f"{k.replace('_L96','').replace('_p0.1','')}: r_E {a:.2f}, r_Δ {-b:.2f}" for k, a, b in rows), fontsize=8.5)
plt.tight_layout(); plt.savefig('/mnt/project-files/figures/fig249_length_from_strength.png', dpi=110); print('ok')
