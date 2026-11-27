"""(374) Рис. 267: стійкість світу в 3D до T300 (L16) і світ на L24 (T100)."""
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = '/mnt/project-files/model/s52/r374'; o = json.load(open(f'{R}/ana374.json'))
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
col = {'0.1': '#2ca02c', '0.2': '#1f77b4'}
for a in ('0.1', '0.2'):
    for s in range(1, 9):
        w = json.load(open(f'{R}/a{a}_seed{s}/t3_s3_L16_s{s}_ell4.json'))['w2']
        ax[0].plot([r['t'] for r in w], [r.get('big', 0)/4096 for r in w], color=col[a], lw=.8, alpha=.8, label=f'α {a}'.replace('.', ',') if s == 1 else None)
ax[0].axvline(100, color='k', ls=':', lw=.8); ax[0].text(103, .05, 'кінець (373)', fontsize=7)
ax[0].set_xlabel('час, од.'); ax[0].set_ylabel('найбільший світ, частка ґратки'); ax[0].set_title('(а) L16, T300: α 0,1 — 8/8 стійкі, α 0,2 — 6/8', fontsize=9); ax[0].legend(fontsize=8)
for r in o['L24']: ax[1].plot(range(1, len(r['big_t'])+1), np.array(r['big_t'])/13824, color='#d62728', lw=.8, alpha=.8, label='L24' if r['seed'] == 1 else None)
for s in range(1, 9):
    w = json.load(open(f'{R}/a0.1_seed{s}/t3_s3_L16_s{s}_ell4.json'))['w2'][:100]
    ax[1].plot([r['t'] for r in w], [r.get('big', 0)/4096 for r in w], color='#2ca02c', lw=.6, alpha=.5, ls='--', label='L16' if s == 1 else None)
ax[1].set_xlabel('час, од.'); ax[1].set_ylabel('найбільший світ, частка ґратки'); ax[1].set_title('(б) α 0,1: частка світу не залежить від розміру (≈0,87)', fontsize=9); ax[1].legend(fontsize=8)
f16 = [r['f300'] for r in o['a0.1']]; f24 = [r['fpos'] for r in o['L24']]
ax[2].scatter(np.full(8, 16) + np.linspace(-.6, .6, 8), f16, color='#2ca02c', s=18, label='частка ψ>0')
ax[2].scatter(np.full(8, 24) + np.linspace(-.6, .6, 8), f24, color='#d62728', s=18)
nw16 = [len(r['neg_wrap']) for r in o['a0.1']]; nw24 = [len(r['big_neg_wrap']) for r in o['L24']]
ax2 = ax[2].twinx(); ax2.bar([15, 23], [np.mean(nw16), np.mean(nw24)], width=1.5, color='#7f7f7f', alpha=.35)
ax2.set_ylabel('залишок Матриці: осей, які він обгортає (сер.)', fontsize=8); ax2.set_ylim(0, 3.2)
ax[2].set_xticks([16, 24]); ax[2].set_xticklabels(['L16 (t300)', 'L24 (t100)']); ax[2].set_ylim(.8, .95); ax[2].set_xlim(12, 28)
ax[2].set_ylabel('частка ґратки у світі'); ax[2].set_title('(в) світ ≈87 %; залишок Матриці (нитки мертві) обгортає тор', fontsize=9)
fig.suptitle('(374) Світ у 3D: Кун, s0 3,0, α 0,1/0,2, по 8 зерен', fontsize=10)
fig.tight_layout(); fig.savefig('/mnt/project-files/figures/fig267_3d_stiikist_L24.png', dpi=130)
