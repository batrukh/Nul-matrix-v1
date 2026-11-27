"""(381) Рис. 274: скан запасу s0 біля краю вікна, α 0,2."""
import json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
o = json.load(open('/mnt/project-files/model/s52/r381/ana/ana381.json'))
col = {2.85: '#1f77b4', 2.9: '#2ca02c', 2.95: '#d62728'}; seen = set()
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
for r in o:
    s = r['s0']; lab = f's0 {s}'.replace('.', ',') if s not in seen else None; seen.add(s)
    ax[0].plot(r['t'], [b/4096 for b in r['big']], color=col[s], lw=1.6 if 'seed8' in r['run'] else .7, ls='--' if 'seed8' in r['run'] else '-', alpha=.85, label=lab)
ax[0].set_xlabel('час, од.'); ax[0].set_ylabel('найбільший світ, частка ґратки'); ax[0].set_title('(а) 21/24: один світ 81–85 %; зерно 8 (штрих) — світу нема', fontsize=9); ax[0].legend(fontsize=8)
for r in o:
    ax[1].scatter(r['fW'], r['AM'], color=col[r['s0']], s=40 if 'seed8' in r['run'] else 20, marker='s' if 'seed8' in r['run'] else 'o')
ax[1].axvspan(0, .18, color='#bbbbbb', alpha=.3); ax[1].text(.01, 1.5, 'бажана зона:\nсвіт ≤ 18–30 %\n+ жива Матриця', fontsize=8)
ax[1].set_xlabel('частка ниток світу'); ax[1].set_ylabel('середнє |z|² у Матриці (A_M)'); ax[1].set_title('(б) або все, або нічого: проміжних станів нема', fontsize=9)
fig.suptitle('(381) Скан запасу s0 2,85–2,95 при α 0,2: Кун L16, 8 зерен на точку, T300', fontsize=10)
fig.tight_layout(); fig.savefig('/mnt/project-files/figures/fig274_3d_skan_zapasu.png', dpi=130)
