"""v1.4 Fig. 13: (a,b) budget window of the variant, (337); (c,d) pinning by random arrows, (349).
Data: data/v14/r335/ana335.json (variant with field, L96, T300), data/v14/r348/ana348.json (no field, gamma=0, s0=3.5, L96)."""
import json, os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
P = os.path.dirname(os.path.abspath(__file__)) + '/../data/v14/'
o = json.load(open(P + 'r335/ana335.json')); A = json.load(open(P + 'r348/ana348.json'))
grp = {}
for k, v in o.items(): grp.setdefault(k.replace('ref_', '').rsplit('_s', 1)[0], []).append(v)
GRN, GRY, ORG, MUT = '#2f8f4e', '#9a9a9a', '#c8642a', '#8a8a8a'
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUT, 'xtick.color': MUT, 'ytick.color': MUT})
fig, ax = plt.subplots(1, 4, figsize=(7.0, 2.1), gridspec_kw={'wspace': 0.55})
S = [3.0, 3.5, 4.0, 4.25, 4.5, 5.5, 6.5, 6.86]
a = ax[0]
a.axvspan(4.0, 4.25, color='0.92', zorder=0)
for s in S:
    f = np.array([v['frac'] for v in grp[f's{s}_L96']]) * 100
    a.plot(np.full(len(f), s) + np.linspace(-0.05, 0.05, len(f)), f, 'o', color=GRN, ms=2.5, alpha=0.7)
    a.plot(s, f.mean(), '_', color='k', ms=8)
a.set_xlabel(r'budget $s_0$'); a.set_ylabel('domain fraction, %')
a = ax[1]
for i, s in enumerate([3.0, 3.5, 4.0, 4.25]):
    n = np.array([v['nworld'] for v in grp[f's{s}_L96']])
    a.plot(np.full(len(n), i) + np.linspace(-0.15, 0.15, len(n)), n, 'o', color=GRN, ms=2.5, alpha=0.7)
    a.errorbar(i, n.mean(), yerr=n.std(), fmt='_', color='k', ms=8, capsize=2, lw=0.8)
a.set_xticks(range(4)); a.set_xticklabels(['3.0', '3.5', '4.0', '4.25']); a.set_xlabel(r'budget $s_0$'); a.set_ylabel('domains at $t=300$')
for k, v in sorted(A.items()):
    if 'lat_ge0' in k: c = ORG
    elif k.startswith('r348/ge0_'): c = GRY
    else: continue
    t = np.array(v['t']); m = t <= 300
    ax[2].plot(t[m], np.array([x['N'] for x in v['fr']])[m], color=c, lw=1.1)
    ax[3].plot(t[m], np.array([x['frac'] for x in v['fr']])[m] * (100 if max(x['frac'] for x in v['fr']) <= 1 else 1), color=c, lw=1.1)
ax[2].set_ylabel('domains $N$'); ax[3].set_ylabel('domain fraction, %')
ax[2].plot([], [], color=ORG, label='lattice arrows'); ax[2].plot([], [], color=GRY, label='random arrows'); ax[2].legend(frameon=False, loc='upper right', fontsize=6.5)
for a in ax[2:]: a.set_xlabel('$t$')
for j, a in enumerate(ax):
    a.spines[['top', 'right']].set_visible(False); a.text(-0.42, 1.04, f'({"abcd"[j]})', transform=a.transAxes, fontweight='bold')
O = os.path.dirname(os.path.abspath(__file__)) + '/../figures/'; os.makedirs(O, exist_ok=True)
fig.savefig(O + 'fig13_window_pinning.pdf', bbox_inches='tight'); fig.savefig(O + 'fig13_window_pinning.png', dpi=200, bbox_inches='tight')
print('ok')
