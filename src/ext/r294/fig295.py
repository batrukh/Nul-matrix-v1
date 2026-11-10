import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
L = 96; fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
ax[0].bar([0, 1, 2], [1.41, 1.064, 1.054], color=['0.6', '0.75', 'C2'], width=0.55)
for i, v in enumerate([1.41, 1.064, 1.054]): ax[0].text(i, v + 0.02, f'{v:.2f}'.replace('.', ','), ha='center')
ax[0].axhline(1/0.985**2, color='k', ls='--', lw=0.8); ax[0].text(2.3, 1.04, 'E₀/c²', fontsize=8, va='bottom')
ax[0].set_xticks([0, 1, 2]); ax[0].set_xticklabels(['κ_a 1, κ_b 1\n(моя помилка, (291))', 'κ_a 0,5, κ_b 1\n(294)', 'κ_a 1, κ_b 2\nБАЗА (295)'], fontsize=8)
ax[0].set_ylabel('інерційна маса / E₀'); ax[0].set_ylim(0, 1.6); ax[0].set_xlim(-0.5, 2.8); ax[0].set_title('Маса спокою в базі: 1,05·E₀', fontsize=9)
def sim(F, m0, c, T):
    v = p = 0.0; tt = 0.0; dt = 0.01; ts, vs = [], []
    while tt < T: gam = 1/np.sqrt(1 - min(v/c, 0.9999)**2); p += F*dt; v = p/np.sqrt(m0**2 + (p/c)**2); tt += dt; ts.append(tt); vs.append(v)
    return np.array(ts), np.array(vs)
for g, col in [('0.01', 'C0'), ('0.02', 'C3')]:
    d = json.load(open(f'mass_ka1_kb2_g{g}.json'))['rec']; t = np.array([q['t'] for q in d]); x = np.unwrap(np.array([[q['p'][i][0] for i in range(2)] for q in d])*2*np.pi/L, axis=0)*L/2/np.pi
    v = np.gradient(x.mean(1), t); ax[1].plot(t, v, 'o', ms=2, color=col, label=f'g {g}'.replace('.', ','))
    F = 1.0571*float(g); ts, vs = sim(F, 1.27, 0.94, 82); ax[1].plot(ts - 2, vs, '-', color=col, lw=1); ax[1].plot(t, F*t/1.27, ':', color=col, lw=1)
ax[1].plot([], [], 'k-', label='з γ (m₀ 1,27, c 0,94)'); ax[1].plot([], [], 'k:', label='без зростання маси')
ax[1].axhline(0.985, color='k', ls='--', lw=0.6); ax[1].set_ylim(0, 1.05); ax[1].set_xlabel('час'); ax[1].set_ylabel('швидкість вихору'); ax[1].legend(fontsize=7, loc='lower right')
ax[1].set_title('Розгін у базі: межа c ≈ 0,9–1,1', fontsize=9)
def spec(t, s, lo=0.25, hi=0.6):
    x = (s - np.polyval(np.polyfit(t, s, 2), t))*np.hanning(len(s)); F = np.abs(np.fft.rfft(x, 64*len(x))); w = 2*np.pi*np.fft.rfftfreq(64*len(x), t[1] - t[0]); m = (w > lo) & (w < hi); return w[m], F[m]/F[m].max()
r = json.load(open('br_ka1_kb2_br_0.1.json'))['rec']; t = np.array([q['t'] for q in r]); s = np.array([q['m2'][0] for q in r]); m = t > 50
w, F = spec(t[m], s[m]); ax[2].plot(w, F, 'k', label='спокій: 0,386')
for g, col, v in [('0.02', 'C0', 0.468), ('0.04', 'C3', 0.648)]:
    A = json.load(open(f'kk_ka1_kb2_g{g}_k0.1.json'))['rec']; B = json.load(open(f'kk_ka1_kb2_g{g}_k0.json'))['rec']; n = min(len(A), len(B)); t = np.array([q['t'] for q in A[:n]])
    s = np.array([q['s'][0][0] for q in A[:n]]) - np.array([q['s'][0][0] for q in B[:n]]); m = (t > 41) & (t < 111)
    w, F = spec(t[m], s[m]); ax[2].plot(w, F, color=col, label=f'v = {v:.2f}'.replace('.', ','))
    ax[2].axvline(0.386*np.sqrt(1 - (v/0.985)**2), color=col, ls='--', lw=0.8)
ax[2].plot([], [], 'k--', lw=0.8, label='0,386/γ, c = 0,985'); ax[2].set_xlabel('частота'); ax[2].set_ylabel('спектр (норм.)'); ax[2].legend(fontsize=7)
ax[2].set_title('Годинник у базі сповільнюється за γ (±0,03)', fontsize=9)
fig.suptitle('(295) Вихор Аліси в базі (κ_a 1, κ_b 2): одна швидкість світла, одна відносність (синтетичний світ L96)', fontsize=10)
fig.tight_layout(); fig.savefig('/mnt/project-files/figures/fig193_base_vortex.png', dpi=130)
