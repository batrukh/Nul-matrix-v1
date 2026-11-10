import json, glob, sys, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, 'code'); import gf104 as GFm
r = json.load(open('ana321.json')); c = json.load(open('core321.json'))
f, ax = plt.subplots(1, 4, figsize=(17, 4.2))
st = {'s1_d0': ('C0', '--', 'зерно 1, без n'), 's1_d1': ('C0', '-', 'зерно 1, з n'), 's2_d0': ('C1', '--', 'зерно 2, без n'), 's2_d1': ('C1', '-', 'зерно 2, з n')}
for d, (col, ls, lab) in st.items():
    t = [q['k']*0.05 for q in r[d]['dtr']]
    ax[0].plot(t, [q['aw'] for q in r[d]['dtr']], color=col, ls=ls, label=lab)
    ax[1].plot(t, [max(q['nz2'], 0.5) for q in r[d]['dtr']], color=col, ls=ls, label=lab + ': Z₂ (Аліса)')
    if d.endswith('d1'): ax[1].plot(t, [max(q['nd'], 0.5) for q in r[d]['dtr']], color=col, ls=':', lw=2.2, label=lab + ': дисклінації n')
ax[0].set(xlabel='час t', ylabel='площа світів (вузли ψ>0,3)', title='а) народження і ріст світів'); ax[0].legend(fontsize=8)
ax[1].set(yscale='log', xlabel='час t', ylabel='трикутники-дефекти у світах', title='б) дефекти: Аліса vs директор'); ax[1].legend(fontsize=7)
d = 's1_d1'; z = np.load(glob.glob(d + '/*_g104.npz')[0]); L = 96
gf = GFm.GF.__new__(GFm.GF); gf.L = L; gf.P, gf.U, gf.n = z['P'], z['U'], z['n']; psi = z['psi'].reshape(L, L)
w, V = np.linalg.eigh(gf.P); sp = w[..., 2] - w[..., 1]
im = ax[2].imshow(np.where(psi > 0.3, sp, np.nan).T, origin='lower', cmap='viridis', vmin=0, vmax=0.14)
dd = gf.ddefects(); det, wm = gf.defects(z['psi'], aware=True)
p = np.argwhere((dd < 0) & wm); ax[2].plot(p[:, 1], p[:, 2], 'o', mfc='none', mec='r', ms=9, mew=2, label='дисклінація n')
q = np.argwhere((det < 0) & wm); ax[2].plot(q[:, 1], q[:, 2], 'x', color='w', ms=8, label='вихор Аліси') if len(q) else None
plt.colorbar(im, ax=ax[2], fraction=0.046, label='розщеплення трійки λ₂−λ₁')
ax[2].set(title='в) зерно 1 з n, t 250: світи і дефекти', xticks=[], yticks=[]); ax[2].legend(fontsize=8, loc='lower right')
for d in ['s1_d1', 's2_d1']:
    for row in c[d]['rows']:
        if not row['z2near']: ax[3].plot(range(9), [x[0] for x in row['ring']], 'r-', alpha=0.5)
ax[3].axhline(np.mean([c[d]['split_world'] for d in c]), color='k', ls='--', label='середнє в світі')
ax[3].set(xlabel='відстань від ядра (кільце, вузли)', ylabel='λ₂−λ₁', title='г) ядро дисклінації: трійка змикається'); ax[3].legend(fontsize=8)
plt.tight_layout(); plt.savefig('/mnt/project-files/figures/fig216_director_birth.png', dpi=110)
