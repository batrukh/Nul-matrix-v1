"""(346) fig246: coarsening with the Phi* field (r346) vs without (r348), with/without elastic brake."""
import json, glob, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
a = json.load(open('ana346.json')); b = json.load(open('../r348/ana348.json'))
G = [(a, 'r346/ge0_L96', 'з Φ★, без гальма', '#c0392b', '-'), (a, 'r346/ge0.0025_L96', 'з Φ★, з гальмом', '#2471a3', '-'),
     (b, 'r348/ge0_L96', 'без Φ★, без гальма', '#c0392b', '--'), (b, 'r348/ge0.0025_L96', 'без Φ★, з гальмом', '#2471a3', '--')]
fig = plt.figure(figsize=(13, 7.6)); gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05])
axN = fig.add_subplot(gs[0, :2]); axM = fig.add_subplot(gs[0, 2:])
for D, pre, lab, c, ls in G:
    ks = sorted(k for k in D if k.startswith(pre + '_s')); t = np.array(D[ks[0]]['t'])
    n = min(len(D[k]['t']) for k in ks); t = t[:n]
    N = np.array([[f['N'] for f in D[k]['fr'][:n]] for k in ks]); M = np.array([[f['mean'] for f in D[k]['fr'][:n]] for k in ks])
    axN.plot(t, N.mean(0), ls, color=c, lw=2, label=f'{lab} ({len(ks)})'); axN.fill_between(t, N.mean(0)-N.std(0), N.mean(0)+N.std(0), color=c, alpha=.08)
    axM.plot(t, M.mean(0), ls, color=c, lw=2, label=lab)
axN.set_xlabel('час t'); axN.set_ylabel('кількість світів N (середнє ± розкид)'); axN.legend(fontsize=9); axN.set_title('а) кількість світів', loc='left')
axM.set_xlabel('час t'); axM.set_ylabel('середня площа світу, вузлів'); axM.set_yscale('log'); axM.set_title('б) середня площа світу', loc='left')
for j, (d, tt) in enumerate([('ge0_L96_s5', 'з Φ★, без гальма, s5'), ('ge0_L96_s3', 'з Φ★, без гальма, s3'), ('ge0.0025_L96_s5', 'з Φ★, з гальмом, s5'), ('ge0.0025_L96_s3', 'з Φ★, з гальмом, s3')]):
    S = np.load(glob.glob(d + '/*_sp.npz')[0]); L = int(round(np.sqrt(S['psi'].shape[1])))
    ax = fig.add_subplot(gs[1, j]); ax.imshow((S['psi'][-1].reshape(L, L) > 0).astype(float), cmap='Greys', vmin=0, vmax=1.4, origin='lower')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f'{"вгде"[j]}) {tt}, t600', loc='left', fontsize=9)
fig.suptitle('(350) Згрубіння світів з полем Φ★ і без нього, L96, T600, база 8.11 (s0 3,5); внизу темне — світ (ψ>0)', fontsize=12)
fig.tight_layout(); fig.savefig('/mnt/project-files/figures/fig247_coarsening_field.png', dpi=130)
