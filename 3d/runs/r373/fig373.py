"""(373) Рис. 266: народження світу в 3D при слабкому α — частка ґратки у найбільшому світі від часу і частка зерен зі світом від α."""
import json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
o = json.load(open('/mnt/project-files/model/s52/r373/ana373.json'))
col = {'0': '#7f7f7f', '0.1': '#2ca02c', '0.2': '#1f77b4'}
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, rows in o.items():
    for i, r in enumerate(rows):
        ax[0].plot(r['t'], [b/4096 for b in r['big']], color=col[a], lw=.8, alpha=.8, label=f'α {a}'.replace('.', ',') if i == 0 else None)
ax[0].set_xlabel('час, од.'); ax[0].set_ylabel('найбільший світ ψ > 0, частка ґратки'); ax[0].set_title('(а) світ народжується з Матриці (по 8 зерен)'); ax[0].legend(fontsize=8)
xs = [0, .1, .2, .3, .4, .5, .7, 1]; fr = [0, 8/8, 4/8, 0, 0, 0, 0, 0]; nn = ['8', '8', '8', '2', '2', '2', '2', '12']
ax[1].bar(range(len(xs)), fr, color=['#7f7f7f', '#2ca02c', '#1f77b4'] + ['#d62728']*5)
for i, n in enumerate(nn): ax[1].text(i, fr[i] + .03, f'n={n}', ha='center', fontsize=7)
ax[1].set_xticks(range(len(xs))); ax[1].set_xticklabels([str(x).replace('.', ',') for x in xs]); ax[1].set_ylim(0, 1.15)
ax[1].set_xlabel('α'); ax[1].set_ylabel('частка зерен зі світом'); ax[1].set_title('(б) вікно α ≈ 0,1–0,2 (червоне — (355), (351))')
fig.suptitle('(373) Перший світ у 3D: Кун L16, s0 3,0, T100, народження з випадкової Матриці (попередньо)', fontsize=10)
fig.tight_layout(); fig.savefig('/mnt/project-files/figures/fig266_3d_narodzhennya_svitu.png', dpi=130)
