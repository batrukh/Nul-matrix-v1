import numpy as np, json, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '../r298'); sys.path.insert(0, '..')
from cgauge import CGauge
plt.rcParams.update({'font.size': 10})
ld = lambda m, d: json.load(open(f'aw_{m}_D{d}.json'))
Ev = -917.015; ea = ld('alice', 0)['E'][-1] - Ev; ew = ld('wind', 16)['E'][-1] - Ev
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2)); cols = {0: 'C3', 4: 'C1', 8: 'C2', 16: 'C0'}
res = {}
for D in (0, 4, 8, 16):
    d = ld('both', D); t = np.array(d['t']); e = np.array(d['E']) - Ev - ea - ew
    ax[0].plot(t, e, c=cols[D], label=f'd = {D}'); res[D] = float(e[-1])
    ec = np.array(d['ec']); bc = np.array(d['bc'])
    ax[1].plot(t, ec[:, 0], c=cols[D], label=f'Аліса, d = {D}')
    if D <= 8: ax[1].plot(t, bc[:, 0], c=cols[D], ls='--')
ax[0].axhline(0, c='k', lw=0.6); ax[0].axvline(100, c='gray', ls=':', lw=0.6)
ax[0].set_ylim(-1.3, 1.0); ax[0].set_xlabel('час t'); ax[0].set_ylabel('E(разом) − E(Аліса) − E(оберт)'); ax[0].legend(fontsize=8)
ax[0].set_title(f'а) енергія зв\'язку: ≈ −0,93 (ядро Аліси {ea/2:.2f})')
ax[1].set_xlabel('час t'); ax[1].set_ylabel('x від місця посадки Аліси'); ax[1].legend(fontsize=7.5, loc='upper right')
ax[1].set_title('б) ядро Аліси (суцільна) і вихор +1 (пунктир)')
f = np.load('aw_both_D8_fin.npz'); P, A, U = f['P'], f['A'], f['U']; L = P.shape[0]
i, j = np.meshgrid(np.arange(L), np.arange(L), indexing='ij'); X = (i + 0.5*j) % L; Y = np.sqrt(3)/2*j
g = CGauge(np.full((L, L), -1.), np.ones((3, L, L)), np.ones((2, L, L))); Hu, Hd = g.hol(U)
so5 = (5 - np.trace(Hu, axis1=-2, axis2=-1)) + (5 - np.trace(Hd, axis1=-2, axis2=-1))
w = lambda x: (x + np.pi) % (2*np.pi) - np.pi; Fu, Fd = g.flux(A); fb = np.abs(w(Fu)) + np.abs(w(Fd))
m = fb > 0.02; ax[2].scatter(X[m], Y[m], c='C3', s=30, alpha=0.35, label='потік U(1) (оберт ±1)')
m2 = so5 > 0.01; ax[2].scatter(X[m2], Y[m2], c='k', s=5, label='потік SO(5) (ядро Аліси)'); ax[2].legend(fontsize=7.5, loc='upper left'); ax[2].set_xlim(0, L); ax[2].set_ylim(0, Y.max())
ax[2].set_aspect('equal'); ax[2].set_xticks([]); ax[2].set_yticks([]); ax[2].set_title('в) d = 8, t = 200: Аліса сіла на +1')
plt.tight_layout(); plt.savefig('/mnt/project-files/figures/fig197_alice_plus_winding.png', dpi=130)
json.dump(dict(E_alice_pair=ea, E_wind_pair=ew, bind=res), open('ana300.json', 'w'), indent=1); print(ea, ew, res)
