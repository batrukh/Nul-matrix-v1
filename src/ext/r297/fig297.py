"""(297) рис.: чи тримає вихор непарний квант. Поштовх лише P (закон Гаусса), поглинальний шар r>16, L96, база."""
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = '/mnt/project-files/model/s52/r297/'
a = np.load(D + 'diff_sp600.npy'); z = np.load(D + 'probe_vort_bg_sp600.npz'); X, Y, cx, cy, dts = z['X'], z['Y'], float(z['cx']), float(z['cy']), float(z['dts'])
t = np.arange(len(a))*dts; zs = np.array([1, 1, -1, -1, 1.]); iu = np.triu_indices(5); ia = np.triu_indices(5, 1)
odd = np.array([zs[i]*zs[j] < 0 for i, j in zip(*iu)] + [zs[i]*zs[j] < 0 for m in range(3) for i, j in zip(*ia)])
th = np.mod(np.arctan2(Y - cy, X - cx) - np.pi/2, 2*np.pi); th = np.where((Y > cy) & (X < cx + 0.25) & (th > np.pi), th - 2*np.pi, th); r = np.hypot(X - cx, Y - cy)
def sp(sel, nu, t1=250):
    w = ((r >= 0.5) & (r < 6))*np.exp(-1j*nu*th); s = t >= t1; A = np.einsum('tnc,n->tc', a[s][:, :, sel], w); A = A - A.mean(0)
    n = 8*len(A); F = np.fft.fft(A*np.hanning(len(A))[:, None], n=n, axis=0)[:n//2]; om = 2*np.pi*np.fft.fftfreq(n, dts)[:n//2]; return om, (np.abs(F)**2).sum(1)
CE, CO = '#2a6fdb', '#d9822b'
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
om, S = sp(~odd, 0); ax[0].semilogy(om, S, color=CE, lw=1.6, label='парні, ν = 0 (контроль)')
for nu, ls in [(0.5, '-'), (1.5, '--')]:
    om, S = sp(odd, nu); ax[0].semilogy(om, S, color=CO, lw=1.6, ls=ls, label=f'непарні, ν = {nu:g}')
for x, lab in [(0.405, 'Φ★ 0,405'), (0.465, '0,465'), (1.007, 'W 1,007')]:
    ax[0].axvline(x, color='0.6', lw=0.8, ls=':'); ax[0].text(x + 0.012, 2e-6, lab, rotation=90, fontsize=9, color='0.4')
ax[0].annotate('дихання 0,386', (0.386, sp(~odd, 0)[1][np.argmin(abs(sp(~odd, 0)[0] - 0.386))]), xytext=(0.1, 3), textcoords='data', fontsize=9, arrowprops=dict(arrowstyle='->', color='0.4'))
ax[0].set_xlim(0, 2.0); ax[0].set_ylim(1e-6, 1e2); ax[0].set_xlabel('частота ω'); ax[0].set_ylabel('потужність у ядрі (t 250–600)')
ax[0].set_title('а) спектр ядра після поштовху'); ax[0].legend(frameon=False, fontsize=9, loc='upper right')
x = a[:, :, odd]; tc, B = [], {k: [] for k in ['0,1–0,95 (нижче порогу W)', '0,95–1,06 (біля порогу W)']}
for t1 in range(50, 501, 50):
    s = (t >= t1) & (t < t1 + 100); y = x[s] - x[s].mean(0); F = np.fft.rfft(y*np.hanning(len(y))[:, None, None], axis=0)
    om_ = 2*np.pi*np.fft.rfftfreq(len(y), dts); P = (np.abs(F)**2).sum((1, 2)); tc.append(t1 + 50)
    B['0,1–0,95 (нижче порогу W)'].append(P[(om_ >= 0.1) & (om_ < 0.95)].sum()); B['0,95–1,06 (біля порогу W)'].append(P[(om_ >= 0.95) & (om_ < 1.06)].sum())
xe = a[:, :, ~odd]; br = []
for t1 in range(50, 501, 50):
    s = (t >= t1) & (t < t1 + 100); y = xe[s] - xe[s].mean(0); F = np.fft.rfft(y*np.hanning(len(y))[:, None, None], axis=0)
    om_ = 2*np.pi*np.fft.rfftfreq(len(y), dts); P = (np.abs(F)**2).sum((1, 2)); br.append(P[(om_ >= 0.37) & (om_ < 0.40)].sum())
ax[1].semilogy(tc, br, 'o-', color=CE, label='парні: дихання 0,37–0,40 (зв\'язане)')
for (k, v), mk in zip(B.items(), ['s-', 'D--']): ax[1].semilogy(tc, v, mk, color=CO, label='непарні: ' + k)
ax[1].set_xlabel('середина вікна, t'); ax[1].set_ylabel('потужність у смузі'); ax[1].set_title('б) що лишається біля вихору з часом')
ax[1].legend(frameon=False, fontsize=9); ax[1].grid(color='0.92')
fig.suptitle('(297) Вихор Аліси (база, L96) тримає парні коливання, але не тримає жодного непарного кванта', fontsize=12)
fig.tight_layout(); fig.savefig('/mnt/project-files/figures/fig195_odd_bound.png', dpi=130)
