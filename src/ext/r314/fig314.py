import numpy as np, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
sys.path.insert(0, '/mnt/project-files/model/s52/r268/code'); import phi90 as PH
d = np.load('flip_vort.npz'); dE, r = d['dE'], d['r']; L = 96; X, Y = PH.coords(L)
D = [(1, 0), (0, 1), (-1, 1)]; cx, cy = 50.0, 24.2487
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for m, dd in enumerate(D):
    mx = X + 0.5*(dd[0] + 0.5*dd[1]); my = Y + 0.5*np.sqrt(3)/2*dd[1]
    s = (abs(mx - cx) < 14) & (abs(my - cy) < 12)
    sc = ax[0].scatter(mx[s], my[s], c=dE[m][s], s=14, cmap='RdBu_r', norm=SymLogNorm(1e-5, vmin=-0.02, vmax=0.02))
ax[0].plot(cx, cy, 'k*', ms=10); ax[0].set_aspect('equal'); plt.colorbar(sc, ax=ax[0], fraction=0.04, label='ΔE перемикання ребра')
ax[0].set_title('(a) ціна перемикання 2→2 біля вихору r294\n(червоне — дорожче, синє — вигідно; у вакуумі 0)')
ok = np.isfinite(dE); ax[1].scatter(r[ok], np.where(dE[ok] > 0, dE[ok], np.nan), s=3, c='C3', label='ΔE > 0')
ax[1].scatter(r[ok], np.where(dE[ok] < 0, -dE[ok], np.nan), s=3, c='C0', label='−ΔE (ΔE < 0)')
ax[1].set_yscale('log'); ax[1].set_ylim(1e-9, 0.1); ax[1].set_xlim(0, 30); ax[1].set_xlabel('r до ядра'); ax[1].set_ylabel('|ΔE|')
ax[1].axhline(1.10, c='k', ls=':'); ax[1].legend(); ax[1].set_title('(b) |ΔE| за радіусом; виграш ≤ 3·10⁻⁴ проти енергії вихору 1,1')
plt.tight_layout(); plt.savefig('fig209_flip_cost.png', dpi=130)
