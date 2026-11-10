import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from eff import *
L, R = 48, 8.0; PAR = dict(t=0.1, rc=4.0, h=1.0, minf=0.333, m3=0.6); c = np.array([0.75*L + 0.25, np.sqrt(3)/4*L + 0.2])
n, U, r, (v1, v2) = config(L, R, 3.0, 0.0, c=c); w, v = low(build(L, n, U, r, **PAR), 8)
x, y = np.meshgrid(np.arange(L), np.arange(L), indexing='ij'); X, Y = x + 0.5*y, np.sqrt(3)/2*y
dens = (v[:, :4]**2).sum(1).reshape(L, L, 2).sum(-1)
phis = np.linspace(0, np.pi, 37); sp_ = []
for ph in phis:
    n, U, r, _ = config(L, R, 3.0, ph, c=c); ww, _ = low(build(L, n, U, r, **PAR), 8); sp_.append(np.sqrt(ww))
sp_ = np.array(sp_)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
sc = ax[0].scatter(X.ravel(), Y.ravel(), c=dens.ravel(), s=9, cmap='magma'); ax[0].set_aspect('equal')
t = np.linspace(0, np.pi, 50); ax[0].plot(c[0] + R*np.cos(t), c[1] + R*np.sin(t), 'c--', lw=1); ax[0].plot(c[0] - R*np.cos(t), c[1] - R*np.sin(t), 'c--', lw=1)
ax[0].plot(*v1, 'c^', ms=8); ax[0].plot(*v2, 'cv', ms=8); ax[0].set_xlim(c[0] - 18, c[0] + 18); ax[0].set_ylim(c[1] - 14, c[1] + 14)
ax[0].set_title('(a) зв\'язані непарні кванти і шлях обміну'); plt.colorbar(sc, ax=ax[0], fraction=0.04)
for k in range(8): ax[1].plot(phis/np.pi, sp_[:, k], 'C3' if k < 4 else 'C7', lw=1.5 if k < 4 else 0.8)
ax[1].axhline(0.333, ls=':', c='k'); ax[1].set_xlabel('φ / π (кут повороту пари)'); ax[1].set_ylabel('ω'); ax[1].set_title('(b) спектр під час обміну: щілина ≈0,05 тримається')
lab = ['непарний квант\n(313) L48', '(313) L64', '(313) L96', 'контроль\nбез Z₂', 'ферміон\n(305)']
data = []
for tag in ['x48', 'x64', 'x96']:
    try: data.append(json.load(open(f'braid_{tag}.json'))['two_one_per_vortex'])
    except Exception: data.append([])
data.append(json.load(open('braid_e48.json'))['two_one_per_vortex']); data.append([0.0, 0.0, 0.0, 1.0])
for i, d in enumerate(data):
    for p in d: ax[2].plot(i + 0.08*(d.index(p) - 1.5), np.cos(np.pi*p), 'o', c='C3' if i < 3 else ('C0' if i == 3 else 'C2'), ms=9, alpha=0.7)
ax[2].set_xticks(range(len(lab))); ax[2].set_xticklabels(lab, fontsize=8); ax[2].set_ylim(-1.2, 1.2); ax[2].axhline(0, c='k', lw=0.5)
ax[2].set_ylabel('cos(фаза обміну), по кванту на вихорі'); ax[2].set_title('(c) обмін: −1 у трьох із чотирьох станів = ферміон')
plt.tight_layout(); plt.savefig('fig208_exchange_odd_quantum.png', dpi=130); print(np.round(sp_[[0, 9, 18, 27, 36]], 4))
