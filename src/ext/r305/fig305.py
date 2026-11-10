import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
A = json.load(open('ana305.json')); T = json.load(open('t1.json'))
fig, ax = plt.subplots(1, 4, figsize=(18, 4.4))
r0s = [k for k in T]; xs = [0, 1, 2, 3, 4, 6, 9]; lab = ['0', '1', '2', '3', '4', '6', '∞']
for x, k in zip(xs, r0s):
    w = np.abs(np.array(T[k])); ax[0].semilogy([x]*len(w), np.maximum(w, 1e-5), 'o', color='C3' if (w < 0.1).any() else 'C0', ms=5)
ax[0].axhline(0.3485, color='k', ls='--', lw=0.8); ax[0].text(5, 0.42, 'край щілини', fontsize=8)
ax[0].set_xticks(xs); ax[0].set_xticklabels(lab); ax[0].set_xlabel('радіус ядра-повороту r0'); ax[0].set_ylabel('|E| 8 найнижчих рівнів')
ax[0].set_title('а) нульові моди є при будь-якому r0 ≥ 1')
runs = [('x48', 'обмін L48 R8'), ('x64', 'обмін L64 R10'), ('x48glob', 'обмін, глобальний'), ('f48', 'повний оберт L48')]
for i, (k, l) in enumerate(runs):
    p = np.array(A[k]['phases1']); ax[1].plot(np.cos(np.pi*p) + 0.0, np.sin(np.pi*p), 'o', ms=9 - 1.5*i, label=l, alpha=0.8)
t = np.linspace(0, 2*np.pi, 200); ax[1].plot(np.cos(t), np.sin(t), 'k', lw=0.5); ax[1].set_aspect('equal')
ax[1].set_title('б) власні значення голономії (1 ферміон)'); ax[1].legend(fontsize=7, loc='center')
ax[1].set_xlim(-1.3, 1.3); ax[1].set_ylim(-1.3, 1.3)
Ul = np.abs(np.array(A['x64']['Ul'])); im = ax[2].imshow(Ul, cmap='Greys', vmin=0, vmax=1)
ax[2].set_xticks(range(4)); ax[2].set_yticks(range(4)); lb = ['v1·1', 'v1·2', 'v2·1', 'v2·2']; ax[2].set_xticklabels(lb); ax[2].set_yticklabels(lb)
ax[2].set_title('в) |U| обміну (L64): моди переходять v1 ↔ v2'); plt.colorbar(im, ax=ax[2], shrink=0.8)
obs = [1, 1, 1, -1]; bos = [1, 1, 1, -1]; fer = [-1, -1, -1, 1]
ax[3].bar(np.arange(4) - 0.25, obs, 0.25, label='обчислено (L48, L64, глоб.)', color='C3')
ax[3].bar(np.arange(4), bos, 0.25, label='бозони з дублетом', color='C2', alpha=0.6)
ax[3].bar(np.arange(4) + 0.25, fer, 0.25, label='ферміони з дублетом', color='C0', alpha=0.6)
ax[3].set_xticks(range(4)); ax[3].set_xticklabels(['трипл.', 'трипл.', 'трипл.', 'сингл.']); ax[3].set_ylabel('фаза обміну (±1)')
ax[3].set_title('г) 2 ферміони, по одному на вихір'); ax[3].legend(fontsize=7)
fig.suptitle('(305) Обмін двох вихорів Аліси з ферміоном на соті: вихор + ферміон = БОЗОН (Z₂-фаза Ааронова–Бома гасить знак Фермі)', fontsize=11)
plt.tight_layout(); plt.savefig('fig201_vortex_exchange.png', dpi=130)
