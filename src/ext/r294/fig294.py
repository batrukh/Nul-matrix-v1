import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
# 1: маса спокою
lab = ['κ_a = 1\n(база)', 'κ_a = 0,5\n(c зрівняні)']
ax[0].bar([0, 1], [1.41, 1.064], color=['0.6', 'C2'], width=0.5)
ax[0].axhline(1/0.985**2, color='k', ls='--', lw=0.8); ax[0].text(1.3, 1.04, 'E₀/c², c = 0,985', fontsize=8, va='bottom')
for i, v in enumerate([1.41, 1.064]): ax[0].text(i, v + 0.02, f'{v:.2f}'.replace('.', ','), ha='center')
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(lab); ax[0].set_ylabel('інерційна маса / E₀'); ax[0].set_ylim(0, 1.6); ax[0].set_xlim(-0.5, 1.9)
ax[0].set_title('Маса спокою: з 1,41·E₀ до 1,06·E₀', fontsize=9)
# 2: v(t) при κ_a 0,5 і 1
d = json.load(open('ana294.json'))['D']; old = json.load(open('../r292/ana292.json'))
for g, col in [('0.01', 'C0'), ('0.02', 'C3'), ('0.04', 'C2')]:
    ax[1].plot(d[g]['t'], d[g]['v'], '-', color=col, lw=1.6, label=f'g {g}'.replace('.', ',') + ', κ_a 0,5')
    ax[1].plot(old[g]['t'], old[g]['v'], ':', color=col, lw=1.2)
ax[1].plot([], [], 'k:', label='κ_a 1 (292)')
ax[1].axhline(0.985, color='k', lw=0.6, ls='--'); ax[1].axhline(0.704, color='0.5', lw=0.6, ls='--')
ax[1].text(1, 0.995, 'c = 0,985 (обидві хвилі)', fontsize=7); ax[1].text(1, 0.715, 'фотон при κ_a 1', fontsize=7, color='0.4')
ax[1].set_ylim(0, 1.05); ax[1].set_xlabel('час'); ax[1].set_ylabel('швидкість вихору'); ax[1].legend(fontsize=7, loc='lower right')
ax[1].set_title('Розгін: межа тепер ≈ 1; при v > 0,6 втрати', fontsize=9)
# 3: оцінки c
rows = [('хвилі Φ★', 0.985, 0.985), ('фотон', 0.704, 1.00), ('зростання маси (γ)', (0.69, 0.77), (0.86, 0.96)), ('годинник: радіальна мода', (0.90, 0.93), (0.94, 0.94)), ('годинник: дихання', (0.94, 0.98), (0.90, 0.90))]
for i, (nm, a, b) in enumerate(rows):
    for val, x0, col in [(a, -0.15, '0.5'), (b, 0.15, 'C2')]:
        lo, hi = (val, val) if np.isscalar(val) else val
        ax[2].plot([i + x0]*2, [lo, hi], color=col, lw=6, solid_capstyle='round'); ax[2].plot(i + x0, (lo + hi)/2, 'o', color=col, ms=5)
ax[2].plot([], [], 'o', color='0.5', label='κ_a = 1 (база)'); ax[2].plot([], [], 'o', color='C2', label='κ_a = 0,5')
ax[2].set_xticks(range(len(rows))); ax[2].set_xticklabels([r[0] for r in rows], rotation=20, fontsize=8); ax[2].set_ylabel('гранична швидкість c'); ax[2].set_ylim(0.6, 1.1)
ax[2].legend(fontsize=8); ax[2].set_title('Усі ознаки сходяться на c ≈ 0,9–1', fontsize=9)
fig.suptitle('(294) Зрівняні швидкості світла (κ_a = 0,5): одна відносність для вихору (синтетичний світ L96)', fontsize=10)
fig.tight_layout(); fig.savefig('/mnt/project-files/figures/fig192_one_c.png', dpi=130)
