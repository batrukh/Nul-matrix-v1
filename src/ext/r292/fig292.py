import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
d = json.load(open('ana292.json')); m0, c, t0 = 1.450, 0.773, 2.0
def simv(F, T, kind='B'):
    v = p = 0.0; tt = 0.0; dt = 0.01; ts, vs = [], []
    while tt < T:
        gam = 1/np.sqrt(1 - min(v/c, 0.9999)**2); p += (F/gam if kind == 'B' else F)*dt; v = p/np.sqrt(m0**2 + (p/c)**2); tt += dt; ts.append(tt); vs.append(v)
    return np.array(ts), np.array(vs)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2)); cols = {'0.01': 'C0', '0.02': 'C3', '0.04': 'C2'}
for g, col in cols.items():
    t, v = np.array(d[g]['t']), np.array(d[g]['v']); F = d[g]['F']
    ax[0].plot(t, v, 'o', ms=2.5, color=col, label=f'g = {g.replace(".", ",")}')
    ts, vs = simv(F, 82); ax[0].plot(ts - t0, vs, '-', color=col, lw=1)
    ax[0].plot(t, F*(t + t0)/1.35 - F*t0/1.35, ':', color=col, lw=1)
    wr = np.array(d[g]['wx'])/np.array(d[g]['wy']); s = t <= (40 if g == '0.04' else 80)
    ax[1].plot(v[s], wr[s]/wr[0], 'o', ms=2.5, color=col)
ax[0].axhline(0.80, color='k', lw=0.6, ls='--'); ax[0].text(1, 0.81, 'світло ґратки 0,80', fontsize=8)
ax[0].plot([], [], 'k:', label='без зростання маси (m = 1,35)'); ax[0].plot([], [], 'k-', label='з γ: m₀ 1,45, c 0,77 (підгонка)')
ax[0].set_ylim(0, 0.9); ax[0].set_xlabel('час'); ax[0].set_ylabel('швидкість вихору'); ax[0].legend(fontsize=7.5, loc='lower right')
ax[0].set_title('Швидкість під сталою силою насичується, як у відносності', fontsize=9)
vv = np.linspace(0, 0.62, 100); ax[1].plot(vv, np.sqrt(1 - (vv/c)**2), 'k-', lw=1, label='1/γ (c = 0,77) — повне сплющення')
ax[1].set_xlabel('швидкість'); ax[1].set_ylabel('ширина вздовж руху / поперек (відносно спокою)'); ax[1].legend(fontsize=8)
ax[1].set_title('Ядро сплющується, але слабше за 1/γ (мірка груба)', fontsize=9)
fig.suptitle('(292) Чи росте маса вихору Аліси зі швидкістю (база b = 0,1, синтетичний світ L96)', fontsize=10)
fig.tight_layout(); fig.savefig('/mnt/project-files/figures/fig190_vortex_gamma.png', dpi=130)
