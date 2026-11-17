import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from cone343 import spectrum
C = json.load(open('cone343.json'))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
a = ax[0]
for key, lab, col in (('n6_tw0', 'пласка, парний квант', '#999999'), ('n5_tw0', 'конус (5), парний', '#2a6fdb'), ('n5_tw1', 'конус (5), непарний (скрут)', '#d9480f'), ('n7_tw1', 'сідло (7), непарний', '#2b8a3e')):
    Rs = [15, 25, 35]; a.plot(Rs, [(C[f'{key}_R{R}']['E'][0] - 0.1)*R*R for R in Rs], 'o-', color=col, label=lab)
a.set_xlabel('радіус ґратки R'); a.set_ylabel('(E₀ − m²)·R²'); a.set_ylim(0, 22); a.legend(fontsize=8)
a.set_title('а) без ями: E₀ → m² як 1/R² — зв\'язаного стану нема;\nнепарний квант дисклінація ще й відштовхує', fontsize=10)
a = ax[1]; Vs = np.linspace(0, 2.0, 21)
for n, tw, lab, col in ((6, 0, 'пласка, парний', '#999999'), (5, 1, 'конус (5), непарний', '#d9480f'), (7, 1, 'сідло (7), непарний', '#2b8a3e')):
    a.plot(Vs, [spectrum(n, 45, tw, V=V)['E'][0] for V in Vs], color=col, label=lab)
a.axhline(0.1, color='k', lw=0.8, ls=':'); a.text(0.05, 0.13, 'поріг континууму m²', fontsize=8)
a.axvline(1.25, color='#d9480f', lw=0.8, ls='--'); a.axvline(1.02, color='#2b8a3e', lw=0.8, ls='--')
a.set_xlabel('глибина ями V у ядрі (кільце ≤ 1)'); a.set_ylabel('E₀ (R = 45)'); a.legend(fontsize=8); a.set_ylim(-0.6, 0.2)
a.set_title('б) непарний квант зв\'язується лише з ямою V ≳ 1,0–1,25\n(парний — за будь-якої ями, як у 2D)', fontsize=10)
plt.tight_layout(); plt.savefig('/mnt/project-files/figures/fig239_disclination_bound.png', dpi=130)
