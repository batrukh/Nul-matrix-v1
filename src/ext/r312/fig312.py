"""(312) рис. 207: директор як справжнє поле (спільна релаксація з вихором, L96)"""
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
E = {t: float(np.load(f'/mnt/project-files/model/s52/r312/fin_{t}.npz')['E']) for t in ['mu001', 'mu001e2', 'mu003', 'mu003e2']}
def low(d):
    for q in json.load(open(f'{d}/m4.json'))['modes']:
        if q['odd'] > 0.5 and q['q2'] > 0.5 and q['w2'] > 1e-5: return q['w'], q['core']
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
a = ax[0]; rc = [4.75, 8, 12, 17]
for d, lab, st in [('R20_001', 'μ 0,01, довільний старт', 's--'), ('R20_001e2', 'μ 0,01, старт «найменше |Φ|»', 'o-'), ('R20_003', 'μ 0,03, довільний старт', 'd--'), ('R20_003e2', 'μ 0,03, старт «найменше |Φ|»', '^-')]:
    p = json.load(open(f'{d}/m4.json'))['prof']; a.plot(rc, [p[k] for k in ['3.5-6', '6-10', '10-14', '14-20']], st, label=lab)
a.set_xlabel('r'); a.set_ylabel('|n·e2|²'); a.set_ylim(-0.05, 1.05); a.legend(fontsize=8); a.grid(alpha=.3); a.set_title('а) два стани директора після спільної релаксації')
b = ax[1]; dE = [E['mu001e2'] - E['mu001'], E['mu003e2'] - E['mu003']]
b.bar(['μ 0,01', 'μ 0,03'], dE, color=['#c0504d' if x > 0 else '#4f81bd' for x in dE]); b.axhline(0, c='k', lw=.8)
for i, x in enumerate(dE): b.text(i, x + (0.01 if x > 0 else -0.03), f'{x:+.3f}', ha='center')
b.set_ylabel('E(e2) − E(парна вісь), уся ґратка'); b.set_title('б) при μ 0,03 стан e2 — основний'); b.grid(alpha=.3, axis='y')
c = ax[2]; labs = ['μ 0,01', 'μ 0,03']; wb = [low('R20_001e2')[0], low('R20_003e2')[0]]; wt = [0.3120, 0.3326]
x = np.arange(2); c.bar(x - 0.18, wb, 0.36, label='вихор, директор e2: зв\'язаний квант'); c.bar(x + 0.18, wt, 0.36, color='#999', label='вакуум: поріг')
for i in range(2): c.text(i - 0.18, wb[i] + 0.005, f'{wb[i]:.3f}', ha='center', fontsize=9); c.text(i + 0.18, wt[i] + 0.005, f'{wt[i]:.3f}', ha='center', fontsize=9)
c.set_xticks(x, labs); c.set_ylim(0, 0.4); c.set_ylabel('найнижча заряджена непарна ω, r<20'); c.legend(fontsize=8, loc='lower right'); c.grid(alpha=.3, axis='y')
c.set_title('в) з живим директором вихор тримає квант')
plt.tight_layout(); plt.savefig('fig207_director_field.png', dpi=130)
print(E, dE)
