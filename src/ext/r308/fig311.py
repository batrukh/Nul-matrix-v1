"""(311) рис. 206: директор із зв'язком μ·nᵀΦ²n сам стає вздовж непарної осі e2 — вихор тримає квант"""
import numpy as np, json, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '../r306'); import frame
m = np.load('R20/meta.npz'); F, Zr, Pref, ref = frame.frames(m); rw, P = m['rw'], m['P0']
pe, pv = np.linalg.eigh(Pref); b3 = pv[:, :3]; ze, zv = np.linalg.eigh(b3.T @ Zr @ b3); b3 = b3 @ zv; eo, e0, e1 = b3[:, 0], b3[:, 1], b3[:, 2]
bins = np.array([3.5, 5, 7, 10, 12, 14, 16, 18, 20]); rc = 0.5*(bins[1:] + bins[:-1])
def prof(f):
    n = np.load(f'R20/{f}.npz')['n']; return [np.mean([(F[k].T @ n[k] @ eo)**2 for k in F if lo <= rw[k] < hi]) for lo, hi in zip(bins[:-1], bins[1:])]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
a = ax[0]
for f, lab, st in [('dir_e2_k0.09_mu1', 'μ·nᵀΦ²n, старт e2', 'o-'), ('dir_e0_k0.09_mu1', 'μ·nᵀΦ²n, старт e0', 's--'), ('dir_e2_k0.09', 'κ·nᵀΦn (308), старт e2', 'd:')]:
    a.plot(rc, prof(f), st, label=lab)
a.set_xlabel('r'); a.set_ylabel('|n·e2|²'); a.set_ylim(-0.05, 1.05); a.legend(fontsize=8); a.grid(alpha=.3); a.set_title('а) куди дивиться директор біля вихору')
b = ax[1]
oo = [np.mean([eo @ F[k].T @ P[k] @ F[k] @ eo for k in F if lo <= rw[k] < hi]) for lo, hi in zip(bins[:-1], bins[1:])]
ee = [np.mean([0.5*(e0 @ F[k].T @ P[k] @ F[k] @ e0 + e1 @ F[k].T @ P[k] @ F[k] @ e1) for k in F if lo <= rw[k] < hi]) for lo, hi in zip(bins[:-1], bins[1:])]
b.plot(rc, oo, 'o-', label='Φ уздовж непарної e2'); b.plot(rc, ee, 's-', label='Φ уздовж парних e0, e1'); b.axhline(-0.3487*0 - 0.349, ls=':', c='k', lw=1, label='вакуум −0,349')
b.set_xlabel('r'); b.set_ylabel('власне значення Φ у рамці'); b.legend(fontsize=8); b.grid(alpha=.3); b.set_title('б) вихор зменшує |Φ| уздовж e2:\nnᵀΦ²n найменше саме там')
c = ax[2]
def low(f):
    for q in json.load(open(f))['modes']:
        if q['odd'] > 0.5 and q['q2'] > 0.5 and q['w2'] > 1e-4: return q['w']
vals = [('r<14', low('R14/m3_dir_e0_k0.09_mu1.json')), ('r<20', low('R20/m3_dir_e0_k0.09_mu1.json')), ('вакуум\nr<20', low('V20/m3_dir_e2_k0.09_mu1.json'))]
c.bar([v[0] for v in vals], [v[1] for v in vals], color=['#4f81bd', '#4f81bd', '#999']); c.axhline(0.3, ls=':', c='k'); c.text(2.2, 0.303, 'm', fontsize=9)
for i, v in enumerate(vals): c.text(i, v[1] + 0.005, f'{v[1]:.3f}', ha='center', fontsize=9)
c.set_ylabel('найнижча заряджена непарна ω'); c.set_title('в) з таким директором вихор\nтримає квант нижче порогу'); c.grid(alpha=.3, axis='y'); c.set_ylim(0, 0.4)
plt.tight_layout(); plt.savefig('fig206_director_mu.png', dpi=130)
