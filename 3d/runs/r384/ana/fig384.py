"""(384) Рис. 275: плато ≈30 % — світ-циліндр на торі; прорив = перехід у шар."""
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
o = [r for r in json.load(open('/mnt/project-files/model/s52/r384/ana/ana384.json')) if r.get('big', 0) > 0]
t = np.array([r['t'] for r in o]); f = np.array([r['big'] for r in o])/4096; w = np.array([sum(r['wrap']) for r in o])
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
cols = ['#cccccc', '#9ecae1', '#fdae6b', '#e6550d']; names = ['куля (не обгортає)', 'циліндр (1 вісь)', 'шар (2 осі)', '3 осі']
for k in range(4):
    m = w == k
    if m.any(): ax[0].axvspan(t[m].min() - 2.5, t[m].max() + 2.5, color=cols[k], alpha=.45, label=names[k])
ax[0].plot(t, f, 'k.-', lw=1)
ax[0].axhline(0.155, ls=':', c='#1f77b4'); ax[0].text(3, .165, 'куля→циліндр 0,155', fontsize=7, color='#1f77b4')
ax[0].axhline(1/np.pi, ls=':', c='#d62728'); ax[0].text(3, .33, 'циліндр→шар 1/π ≈ 0,318', fontsize=7, color='#d62728')
ax[0].set_xlabel('час, од.'); ax[0].set_ylabel('світ, частка вузлів'); ax[0].set_ylim(0, .9); ax[0].legend(fontsize=7, loc='upper left', bbox_to_anchor=(0, .9))
ax[0].set_title('(а) плато = світ-циліндр; прорив = перехід у шар', fontsize=9)
ax[1].plot(t, [r['Aedge'] for r in o], label='|z|² на межі світу'); ax[1].plot(t, [r['Afar'] for r in o], label='|z|² у далекій Матриці')
ax2 = ax[1].twinx(); ax2.plot(t, [r['dead_edge'] for r in o], 'r--', label='мертві нитки на межі'); ax2.set_ylabel('частка мертвих (|z|² < 0,01)', color='r')
ax[1].set_xlabel('час, од.'); ax[1].set_ylabel('середнє |z|²'); ax[1].legend(fontsize=7, loc='lower left'); ax2.legend(fontsize=7, loc='upper right')
ax[1].set_title('(б) Матриця виснажується; мертвих ниток на плато лише ×1,3', fontsize=9)
fig.suptitle('(384) Плато ≈30 %: повтор (381) s0 2,95, α 0,2, зерно 7, знімки кожні 5 од. (попередньо, 1 зерно)', fontsize=10)
fig.tight_layout(); fig.savefig('/mnt/project-files/figures/fig275_3d_plato_tsylindr.png', dpi=130)
