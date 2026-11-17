import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
o = json.load(open('ana335.json')); C = json.load(open('coexist337.json'))
grp = {}
for k, v in o.items(): grp.setdefault(k.replace('ref_', '').rsplit('_s', 1)[0], []).append(v)
S = [3.0, 3.5, 4.0, 4.25, 4.5, 5.5, 6.5, 6.86]
fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))
a = ax[0, 0]
for s in S:
    vs = grp[f's{s}_L96']; f = np.array([v['frac'] for v in vs])*100
    a.plot(s + np.linspace(-0.04, 0.04, len(f)), f, 'o', color='#1f77b4', alpha=0.6, ms=5)
    a.errorbar(s, f.mean(), yerr=f.std(), fmt='_', color='k', ms=18, capsize=4)
    a.text(s, f.max() + 4, f'n={len(f)}', ha='center', fontsize=8)
a.axvspan(3.3, 4.2, color='0.9', zorder=0, label='вікно в статті v1 (3,3–4,2)'); a.legend(fontsize=8, loc='upper right')
a.set_xlabel('запас s₀'); a.set_ylabel('частка світу в t300, %'); a.set_title('(а) вікно світів: край між 4,0 і 4,25 різкий')
a = ax[0, 1]
for i, s in enumerate([3.0, 3.5, 4.0, 4.25]):
    vs = grp[f's{s}_L96']; n = np.array([v['nworld'] for v in vs])
    a.plot(np.full(len(n), i) + np.linspace(-0.12, 0.12, len(n)), n, 'o', color='#d62728', alpha=0.7)
    a.errorbar(i, n.mean(), yerr=n.std(), fmt='_', color='k', ms=18, capsize=4)
    born = sum(1 for v in vs if v['first'] is not None)
    a.text(i, -1.2, f'народження {born}/{len(vs)}', ha='center', fontsize=8)
a.set_xticks(range(4)); a.set_xticklabels(['3,0', '3,5', '4,0', '4,25']); a.set_ylim(-1.8, 8)
a.set_xlabel('запас s₀'); a.set_ylabel('кількість світів (L96)'); a.set_title('(б) при 3,0 світи частіше зливаються в один'); a.text(1.15, 7.4, 's₀ 3,5: s5–8 у t600', fontsize=8)
a = ax[1, 0]
L96 = grp['s3.5_L96']; L192 = grp['s3.5_L192']
keys = [('per1000', 'світів на 1000 вузлів', 1), ('frac', 'частка світу', 1), ('depth', 'глибина /10', 0.1)]
for j, (k, lab, sc) in enumerate(keys):
    for dx, vs, c in ((-0.18, L96, '#1f77b4'), (0.18, L192, '#9467bd')):
        y = np.array([v[k] for v in vs])*sc
        a.bar(j + dx, y.mean(), 0.34, color=c, alpha=0.6, yerr=y.std(), capsize=3)
        a.plot(np.full(len(y), j + dx), y, 'k.', ms=4)
a.bar(0, 0, color='#1f77b4', alpha=0.6, label=f'L96 (n={len(L96)})'); a.bar(0, 0, color='#9467bd', alpha=0.6, label=f'L192 (n={len(L192)})')
a.set_xticks(range(3)); a.set_xticklabels([k[1] for k in keys]); a.legend(fontsize=8); a.set_title('(в) s₀ 3,5: L96 проти L192')
a = ax[1, 1]
for k, v in C.items():
    c = '#2ca02c' if k.startswith('r276') else '#ff7f0e'
    a.plot([300, 600], [v['nB300']/9216*100, v['nB600']/9216*100], 'o-', color=c, alpha=0.8)
    yo = {'r335_s5': 0.5, 'r276_s2': -0.5}.get(k, 0)
    a.text(612, v['nB600']/9216*100 + yo, f"{v['nw600']} св." + (' +вихор' if v['vort'] else ''), fontsize=8, va='center')
a.plot([], [], 'o-', color='#2ca02c', label='r276 s1–4'); a.plot([], [], 'o-', color='#ff7f0e', label='нові s5–8'); a.legend(fontsize=8, loc='lower left'); a.set_xlim(250, 700); a.set_xticks([300, 600]); a.set_xlabel('t'); a.set_ylabel('площа світів, %')
a.set_title('(г) s₀ 3,5, T600: 8/8 зберігають 3–7 світів')
plt.suptitle('(337) Нічна пачка статистики, база 8.11: 44 прогони', fontsize=12); plt.tight_layout()
plt.savefig('/mnt/project-files/figures/fig233_stat_batch.png', dpi=110)
