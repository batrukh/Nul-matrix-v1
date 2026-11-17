"""v1.1 (review): (a) twin runs -> Lyapunov exponent; (b) rotation alpha scan; (c) arrow correlation length scan. Data: model/s52/r353."""
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
Y = json.load(open('/mnt/project-files/model/s52/r353/lyap/lyap353.json')); A = json.load(open('/mnt/project-files/model/s52/r353/ana353.json'))
B = json.load(open('/mnt/project-files/model/s52/r348/ana348.json'))
def last(d, k):
    v = d[k]; i = int(np.argmin(np.abs(np.array(v['t']) - 300))); return v['fr'][i]
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
for k, lab, c in (('ge0.0025_L96_s1', r'$\gamma=0{,}0025$', 'C0'), ('ge0_L96_s1', r'$\gamma=0$', 'C1')):
    t = np.array(Y[k]['t']) - 590; ax[0].semilogy(t, Y[k]['dz'], 'o-', ms=3, c=c, label=lab + ', нитки')
    ax[0].semilogy(t, np.maximum(Y[k]['sgnflip'], 1e-5), 's--', ms=3, c=c, alpha=.6, label=lab + ', вузли доменів')
tt = np.linspace(2, 14, 10); ax[0].semilogy(tt, 1e-4*np.exp(0.45*(tt - 2)), 'k:', label=r'$e^{0.45t}$')
ax[0].set(xlabel='час після збурення', ylabel='с.к. різниця ниток / частка', title=r'(а) двійники, збурення $10^{-8}$'); ax[0].legend(fontsize=7)
ks = sorted({k for k in B if 'ge0.0025_L96' in k and k.startswith('r348/')})
al = [0, 0.25, 0.5, 0.75, 1.0]; Nm, Fm = [], []
for a in al:
    keys = [f'r353/a{a:g}_ge0.0025_s{s}' for s in (1, 2)] if a < 1 else ks
    fr = [last(A if a < 1 else B, k) for k in keys]
    ax[1].plot([a]*len(fr), [f['frac'] for f in fr], 'o', c='C0', alpha=.7); ax2 = ax[1]
    Nm.append(np.mean([f['N'] for f in fr]))
a2 = ax[1].twinx(); a2.plot(al, Nm, 's-', c='C3'); a2.set_ylabel('кількість світів (середнє)', color='C3')
ax[1].set(xlabel=r'обертання $\alpha$', ylabel='частка світів (точки)', title=r'(б) обертання, $\gamma=0{,}0025$, $t=300$')
lt = {8: [f'r353/l8_ge0_s{s}' for s in (1, 2)], 16: sorted(k for k in B if k.startswith('r348/ge0_L96')), 32: [f'r353/l32_ge0_s{s}' for s in (1, 2)]}
a3 = ax[2].twinx(); Nl = []
for l, keys in lt.items():
    fr = [last(A if l != 16 else B, k) for k in keys]
    ax[2].plot([l]*len(fr), [f['frac'] for f in fr], 'o', c='C1', alpha=.7); Nl.append(np.mean([f['N'] for f in fr]))
a3.plot(list(lt), Nl, 's-', c='C3'); a3.set_ylabel('кількість світів (середнє)', color='C3'); a3.set_ylim(0, 9)
ax[2].set_xscale('log', base=2); ax[2].set_xticks([8, 16, 32]); ax[2].set_xticklabels(['8', '16', '32']); ax[2].set_ylim(0, 0.6)
ax[2].set(xlabel=r'довжина кореляції стріл $\ell_\tau$', ylabel='частка світів (точки)', title=r'(в) безлад стріл, $\gamma=0$, $t=300$')
fig.tight_layout(); fig.suptitle('(353) рецензія статті про домени: хаос ниток, обертання, безлад стріл', fontsize=11); fig.tight_layout(); fig.savefig('/mnt/project-files/figures/fig252_domeny_recenzia.png', dpi=130)
