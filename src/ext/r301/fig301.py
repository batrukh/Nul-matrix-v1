import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = '/mnt/project-files/model/s52/r301/'; b = json.load(open(D + 'band301.json')); a = json.load(open(D + 'ana301.json'))
plt.rcParams.update({'font.size': 10}); fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
c = b['comp_odd']; t = [r['t'] for r in c]
ax[0].semilogy(t, [r['b053'] for r in c], 'o-', c='C3', label='заряджена Аліса: лінія 0,54 (непарні)')
ax[0].semilogy(t, [r['b053'] for r in b['comp_even']], 'o--', c='C3', alpha=0.5, label='та сама лінія, парні')
ax[0].semilogy(t, [r['b07'] for r in c], 's-', c='C0', label='заряджена Аліса: смуга 0,69–0,74')
ax[0].semilogy(t, [r['b076'] for r in b['alice_odd']], '^-', c='C2', label='лише Аліса: смуга 0,74–0,80')
ax[0].semilogy(t, [r['b053'] for r in b['alice_odd']], 'v-', c='C2', alpha=0.5, label='лише Аліса: 0,50–0,55')
ax[0].set_xlabel('середина вікна t'); ax[0].set_ylabel('сила смуги (поштовх − фон)'); ax[0].legend(fontsize=7); ax[0].set_title('а) лінія 0,54 виходить на плато')
for key, lab, col in (('comp_odd_late', 'заряджена Аліса, t 100–400', 'C3'), ('alice_odd_late', 'лише Аліса, t 100–400', 'C2')):
    om, S = np.array(a[key]['om']), np.array(a[key]['S']); m = om > 0.2; ax[1].semilogy(om[m], S[m]/S[m].max(), c=col, label=lab)
om, S = np.array(a['vac_odd']['om']), np.array(a['vac_odd']['S']); m = om > 0.2; ax[1].semilogy(om[m], S[m]/S[m].max(), c='gray', lw=0.8, label='вакуум (k≈0): пороги')
for th in (0.273, 0.546, 0.702): ax[1].axvline(th, c='k', ls=':', lw=0.7)
ax[1].set_xlim(0.2, 1.2); ax[1].set_ylim(1e-6, 2); ax[1].set_xlabel('ω'); ax[1].legend(fontsize=7.5); ax[1].set_title('б) непарний спектр у ядрі')
k = np.load(D + 'probe_comp.npz'); bg = np.load(D + 'probe_comp_bg.npz'); x = k['rec'] - bg['rec']; dts = float(k['dts']); tt = np.arange(len(x))*dts
s = (tt >= 200) & (tt < 400); y = x[s]; n = len(y); tl = np.arange(n); Y2 = y.reshape(n, -1); C = np.polynomial.polynomial.polyfit(tl, Y2, 2); Y2 = (Y2 - np.polynomial.polynomial.polyval(tl, C).T).reshape(y.shape)
ph = np.exp(-1j*0.54*tl*dts)*np.hanning(n); amp = np.sum(np.abs(np.einsum('t,tpc->pc', ph, Y2))**2 + np.abs(np.einsum('t,tpc->pc', np.conj(ph), Y2))**2, 1)
im = ax[2].scatter(k['dx'], k['dy'], c=amp/amp.max(), s=60, cmap='magma'); plt.colorbar(im, ax=ax[2], label='сила лінії 0,54 (відн.)')
ax[2].set_aspect('equal'); ax[2].set_xlabel('x від ядра'); ax[2].set_ylabel('y від ядра'); ax[2].set_title('в) мода сидить у ядрі (78 % при r < 4)')
plt.tight_layout(); plt.savefig('/mnt/project-files/figures/fig198_charged_alice_mode.png', dpi=130); print('ok')
