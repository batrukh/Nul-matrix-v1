import json, glob, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
r = json.load(open('ana335.json'))
def s0(k): return 3.5 if k.startswith('ref') else float(k.split('_')[0][1:])
L96 = {k: v for k, v in r.items() if v['L'] == 96}
fig, ax = plt.subplots(2, 2, figsize=(12, 9))
a = ax[0, 0]; b = a.twinx()
for k, v in L96.items():
    a.plot(s0(k), 100*v['frac'], 'o', color='C0', ms=7, alpha=0.8); b.plot(s0(k) + 0.04, v['nworld'], 's', color='C3', ms=5, alpha=0.8)
a.set_xlabel('запас інтенсивності s₀'); a.set_ylabel('частка світу в t300, %', color='C0'); b.set_ylabel('кількість світів (L96)', color='C3')
a.axvspan(3.3, 4.2, color='0.9', zorder=0, label='вікно в статті v1 (3,3–4,2)'); a.axvline(6.86, color='k', ls=':', lw=1)
a.text(6.8, 25, 'природний\nзапас 6,86', fontsize=8, ha='right'); a.legend(loc='upper right', fontsize=8)
a.set_title('(а) Вікно світів на базі 8.11: світи лише при s₀ ≤ 4,0', fontsize=10)
a = ax[0, 1]
for k, v in L96.items():
    a.plot(s0(k), v['boil'], 'o', color='C1', ms=7, alpha=0.8); a.plot(s0(k), v['hol'], '^', color='C2', ms=6, alpha=0.8)
a.plot([], [], 'o', color='C1', label='тремтіння: ⟨|ΔA²|⟩ за од. часу, t250–300'); a.plot([], [], '^', color='C2', label='безлад фаз |гол.|')
a.axvline(6.86, color='k', ls=':', lw=1); a.axvspan(6.3, 6.5, color='C1', alpha=0.12, label='поріг кипіння в статті (стріли ґратки)')
a.set_ylim(0, 0.8); a.set_xlabel('запас s₀'); a.legend(fontsize=8); a.set_title('(б) Матриця кипить при всіх запасах, і при природному 6,86', fontsize=10)
a = ax[1, 0]
keys = [('s3.5 L96', [v for k, v in r.items() if k.startswith('ref')]), ('s3.5 L192', [v for k, v in r.items() if 'L192' in k])]
mets = [('per1000', 'світів на 1000 вузлів'), ('frac', 'частка світу'), ('depth', 'глибина /10'), ('maxworld', 'найбільший, тис. вузлів')]
for j, (m, lab) in enumerate(mets):
    for i, (nm, vs) in enumerate(keys):
        sc = 0.1 if m == 'depth' else (1e-3 if m == 'maxworld' else 1)
        vals = [v[m]*sc for v in vs]; x = j + (i - 0.5)*0.35
        a.bar(x, np.mean(vals), 0.33, color=('C0' if i == 0 else 'C4'), alpha=0.6, label=nm if j == 0 else None)
        a.plot([x]*len(vals), vals, 'k.', ms=8)
a.set_xticks(range(len(mets))); a.set_xticklabels([q[1] for q in mets], fontsize=8); a.legend(fontsize=8)
a.set_title('(в) L96 (4 реалізації) проти L192 (2) при s₀ 3,5, t300', fontsize=10)
a = ax[1, 1]; a.axis('off')
a.text(0, 0.5, 'База 8.11 (P 0,16, b 0,1, β 1,3, стріли ℓ_τ 16), T300, по 2 реалізації\n\n'
       '• Верхній край вікна ≈ 4,0–4,25: при 4,0 світи пізно (t 16–22)\n  і малі (2–13 %), при 4,25 і вище — жодного (як 4,2 у статті).\n'
       '• Нижнього краю нема до 3,0: світи 50–58 %, але їх менше (3–5),\n  а Матриця між ними збіднена (натяг 0,9–1,3 проти 2,1–2,7).\n'
       '• Кипіння є при всіх s₀ до природного 6,86: його запускають\n  випадкові стріли (як (51)); поріг 6,3–6,5 статті — лише для\n  стріл ґратки.\n'
       '• L192: світів на 1000 вузлів 0,33–0,46 проти 0,43–0,76 на L96,\n  частка та сама (39–43 % проти 37–53 %), світи глибші (14–16 проти\n  10–11). Густина від L майже не залежить (у межах ×1,6).', fontsize=9.5, va='center')
fig.suptitle('(335) Ключові числа v1 на базі 8.11: вікно запасу, кипіння, розмір Матриці', fontsize=11)
plt.tight_layout(); plt.savefig('/mnt/project-files/figures/fig231_base_window.png', dpi=115)
