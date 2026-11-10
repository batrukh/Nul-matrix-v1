import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
def spec(t, s, lo=0.2, hi=0.6):
    x = (s - np.polyval(np.polyfit(t, s, 2), t))*np.hanning(len(s)); dt = t[1] - t[0]
    F = np.abs(np.fft.rfft(x, 64*len(x))); w = 2*np.pi*np.fft.rfftfreq(64*len(x), dt); m = (w > lo) & (w < hi); return w[m], F[m]/F[m].max()
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
r = json.load(open('mv_g0_acc0.json'))['rec']; t = np.array([q['t'] for q in r]); s = np.array([q['s'][0][0] for q in r]); m = t > 5
w, F = spec(t[m], s[m]); ax[0].plot(w, F, 'k', label='спокій: ω₀ = 0,384')
for g, col, v in [('0.02', 'C0', 0.359), ('0.04', 'C3', 0.492)]:
    A = json.load(open(f'kk_g{g}_k0.1.json'))['rec']; B = json.load(open(f'kk_g{g}_k0.json'))['rec']; n = min(len(A), len(B)); t = np.array([q['t'] for q in A[:n]])
    s = np.array([q['s'][0][0] for q in A[:n]]) - np.array([q['s'][0][0] for q in B[:n]]); m = t > 41
    w, F = spec(t[m], s[m]); ax[0].plot(w, F, color=col, label=f'v = {v:.2f}'.replace('.', ','))
    for c, ls in [(0.70, ':'), (1.0, '--')]: ax[0].axvline(0.384*np.sqrt(1 - (v/c)**2), color=col, ls=ls, lw=0.8)
ax[0].axvline(0.404, color='0.5', lw=0.6); ax[0].text(0.405, 0.9, 'поріг 0,404', fontsize=7)
ax[0].plot([], [], 'k:', lw=0.8, label='ω₀/γ, c = 0,70'); ax[0].plot([], [], 'k--', lw=0.8, label='ω₀/γ, c = 1')
ax[0].set_xlabel('частота'); ax[0].set_ylabel('спектр (норм.)'); ax[0].legend(fontsize=7); ax[0].set_title('«Дихання» ядра у спокої і в русі', fontsize=9)
vv = np.linspace(0, 0.6, 100)
for c, ls in [(0.70, ':'), (1.0, '--')]: ax[1].plot(vv, np.sqrt(1 - (vv/c)**2), 'k' + ls, lw=1, label=f'1/γ, c = {c:.2f}'.replace('.', ','))
ax[1].plot([0, 0.359, 0.492], [1, 0.357/0.384, 0.327/0.384], 'o', color='C2', label='дихання (0,384)')
ax[1].plot([0, 0.359, 0.492], [1, 1.300/1.418, 1.19/1.418], 's', color='C1', label='радіальна мода (1,42)')
ax[1].set_xlabel('швидкість вихору'); ax[1].set_ylabel('частота в русі / у спокої'); ax[1].legend(fontsize=7); ax[1].set_ylim(0.65, 1.02)
ax[1].set_title('«Годинник» сповільнюється, як для c ≈ 0,9–1', fontsize=9)
d = np.load('iso.npz'); modes = d['modes']; rec = d['rec']; L = 48; dt = 0.25; tt = np.arange(len(rec))*dt
for k, (n, mm) in enumerate(modes):
    kx = 2*np.pi*n/L; ky = (2*np.pi*mm/L - 0.5*kx)/(np.sqrt(3)/2); kk = np.hypot(kx, ky); ang = round(np.degrees(np.arctan2(ky, kx)))
    x = rec[:, k]*np.hanning(len(tt)); FF = np.abs(np.fft.fft(x, 16*len(x))); w = 2*np.pi*np.fft.fftfreq(16*len(x), dt); sel = (np.abs(w) < 2) & (np.abs(w) > 0.05)
    i = np.argmax(FF*sel)
    if abs(w[i]) < 0.06 or abs(w[i])/kk > 1.2: continue
    typ = -30 if mm == 0 else (90 if n == 0 else 0)
    ax[2].plot(kk, abs(w[i]), 'o', color={-30: 'C0', 0: 'C3', 90: 'C2'}[typ], ms={-30: 9, 0: 5, 90: 5}[typ])
for a_, col in [(-30, 'C0'), (0, 'C3'), (90, 'C2')]: ax[2].plot([], [], 'o', color=col, label=f'напрям {a_}°')
kq = np.linspace(0, 0.95, 10); ax[2].plot(kq, 0.704*kq, 'k-', lw=0.8, label='ω = 0,704·k'); ax[2].plot(kq, kq, 'k--', lw=0.8, label='ω = k (хвилі Φ★, теорія)')
ax[2].set_xlabel('хвильове число k'); ax[2].set_ylabel('частота «фотона»'); ax[2].legend(fontsize=7); ax[2].set_title('«Фотон» ізотропний, c = 0,70 (не 0,80)', fontsize=9)
fig.suptitle('(293) Годинник вихору в русі і швидкість «світла» (база b = 0,1, синтетичний світ L96)', fontsize=10)
fig.tight_layout(); fig.savefig('/mnt/project-files/figures/fig191_vortex_clock.png', dpi=130)
