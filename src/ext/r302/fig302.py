import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = '/mnt/project-files/model/s52/r302/'; a = json.load(open(D + 'ana302.json')); W = a['windows']
t = np.array([w['t'] for w in W]); plt.rcParams.update({'font.size': 10}); fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
fits = {}
for key, lab, c in (('odd_b054', 'непарні, лінія 0,50–0,56', 'C3'), ('odd_b07', 'непарні, 0,68–0,74', 'C0'), ('odd_b08', 'непарні, 0,74–0,90', 'C2'), ('even_b054', 'парні, 0,50–0,56', 'gray')):
    y = np.array([w[key] for w in W]); ax[0].semilogy(t, y, 'o-', c=c, label=lab)
    p = np.polyfit(t, np.log(y), 1); fits[key] = float(-1/p[0])
ax[0].set_xlabel('середина вікна t (вікно 200)'); ax[0].set_ylabel('сила смуги в ядрі r < 6'); ax[0].legend(fontsize=7.5)
ax[0].set_title(f'а) лінія 0,536 згасає повільно: τ_E ≈ {fits["odd_b054"]:.0f}')
k = np.load(D + 'probe302_kick.npz'); b = np.load(D + 'probe302_bg.npz'); x = k['rec'] - b['rec']; dts = float(k['dts']); tt = np.arange(len(x))*dts
zs = np.array([1, 1, -1, -1, 1.]); iu = np.triu_indices(5); odd = np.array([zs[i]*zs[j] < 0 for i, j in zip(*iu)]); r = np.hypot(k['dx'], k['dy'])
for (t1, t2), c in (((0, 200), 'C1'), ((500, 1000), 'C3')):
    s = (tt >= t1) & (tt < t2); y = x[s][:, r < 6][:, :, odd]; n = len(y); tl = np.arange(n); Y2 = y.reshape(n, -1)
    C = np.polynomial.polynomial.polyfit(tl, Y2, 2); Y2 = Y2 - np.polynomial.polynomial.polyval(tl, C).T
    N = 16*n; F = (np.abs(np.fft.fft(Y2*np.hanning(n)[:, None], n=N, axis=0)[:N//2])**2).sum(1); om = 2*np.pi*np.fft.fftfreq(N, dts)[:N//2]
    m = (om > 0.2) & (om < 1.2); ax[1].semilogy(om[m], F[m]/F[m].max(), c=c, label=f'непарні, t {t1}–{t2}')
for th in (0.273, 0.546, 0.702): ax[1].axvline(th, c='k', ls=':', lw=0.7)
ax[1].set_xlabel('ω'); ax[1].set_ylim(1e-5, 2); ax[1].legend(fontsize=8); ax[1].set_title('б) спектр у ядрі; пунктир — пороги вакууму')
q = a['ang']; nus = np.array(q['(1.0, 3.0)_odd']['nu']); w = 0.25
for o, (ring, c) in enumerate((('(1.0, 3.0)', 'C3'), ('(3.0, 5.5)', 'C1'), ('(5.5, 9.0)', 'C0'))):
    ax[2].bar(nus + (o - 1)*w*0.9/1.5*0.6, q[f'{ring}_odd']['p'], width=0.14, color=c, label=f'r {ring[1:-1].replace(", ", "–")}: півцілі {q[f"{ring}_odd"]["half"]:.0%}')
ax[2].set_xlabel('кутове число ν (θ від розрізу Аліси)'); ax[2].set_ylabel('частка'); ax[2].legend(fontsize=8); ax[2].set_title('в) кутова структура лінії: ν не визначене')
plt.tight_layout(); plt.savefig('/mnt/project-files/figures/fig199_charged_alice_long.png', dpi=130)
json.dump(fits, open(D + 'fits302.json', 'w')); print(fits)
