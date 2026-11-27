"""fig301: (449) перекриття острова з еталоном (425) для 4 τ-зерен × 8 шумів; сірим — 95-й процентиль нуля (зсуви)."""
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
r = json.load(open('ana449b.json')); S = [5, 6, 7, 12]
fig, ax = plt.subplots(figsize=(6.4, 4), dpi=150)
for i, s in enumerate(S):
    x = [q for q in r if q['seed'] == s]
    ov = [q['overlap'] for q in x]; nu = [q['null95'] for q in x]
    xs = i + np.linspace(-0.25, 0.25, len(x))
    ax.bar(xs, nu, width=0.06, color='0.75', label='випадковий зсув (95 %)' if i == 0 else None)
    ax.plot(xs, ov, 'o', color='C0', label='острів (449) з новим шумом' if i == 0 else None)
ax.axhline(0.5, ls='--', c='k', lw=0.8, label='поріг «те саме місце» 0,5')
ax.set_xticks(range(4)); ax.set_xticklabels([f'τ-зерно {s}' for s in S]); ax.set_ylim(0, 1.3)
ax.set_ylabel('перекриття з островом (425)'); ax.set_title('(449) Місце острова задає поле стріл τ: 32/32')
ax.legend(loc='upper center', fontsize=8, ncol=2); fig.tight_layout(); fig.savefig('/mnt/project-files/figures/fig301_3d_ostriv_misce_tau.png')
