import numpy as np, json, glob, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for i, (D, R) in enumerate([('R14', 14), ('R20', 20)]):
    ms = json.load(open(f'{D}/modes_a0.0_r3.0.json'))['modes']
    w = np.array([q['w'] for q in ms]); odd = np.array([q['odd'] for q in ms]) > 0.5; core = np.array([q['core'] for q in ms])
    sel = w < 0.45
    ax[0].scatter(np.full(sel.sum(), i*2) + 0.25*odd[sel] - 0.125, w[sel], c=np.where(odd[sel], 'C3', 'C0'), s=20 + 200*core[sel], alpha=0.6)
vac = {'R14': 0.2413, 'R20': 0.1685}
for i, D in enumerate(['R14', 'R20']): ax[0].hlines(vac[D], i*2 - 0.4, i*2 + 0.4, color='k', ls='--', lw=1)
ax[0].set_xticks([0, 2]); ax[0].set_xticklabels(['область r<14', 'область r<20']); ax[0].set_ylabel('ω')
ax[0].set_title('а) моди біля вихору: червоні — Z-непарні, сині — парні\n(розмір кружка — частка в ядрі r<4; пунктир — найнижча мода без вихору)', fontsize=9)
Rs = np.array([14, 20]); wodd = np.array([0.1601, 0.1113]); wvac = np.array([0.2413, 0.1685])
ax[1].plot(1/Rs, wodd, 'o-', color='C3', label=f'непарні з вихором, ωR = {np.mean(wodd*Rs):.2f}')
ax[1].plot(1/Rs, wvac, 's--', color='k', label=f'вакуум, ωR = {np.mean(wvac*Rs):.2f}')
ax[1].plot([0, 0.075], [0, 0], 'k', lw=0.4); ax[1].set_xlim(0, 0.075); ax[1].set_ylim(0, 0.26)
ax[1].set_xlabel('1/R'); ax[1].set_ylabel('найнижча ω'); ax[1].legend(fontsize=8); ax[1].set_title('б) безмасові: ω → 0 як 1/R (зв\'язаного стану з ω>0 нема)', fontsize=9)
for D, c in [('R14', 'C3'), ('R20', 'C1')]:
    pts = []
    for f in glob.glob(f'{D}/modes_a*_r3.0.json'):
        d = json.load(open(f)); m0 = [q for q in d['modes'] if q['odd'] > 0.5][0]; pts.append((d['alpha'], m0['w2']))
    pts = sorted(pts); a_, w2 = np.array(pts).T; ax[2].plot(a_, w2, 'o-', color=c, label=D.replace('R', 'r<'))
ax[2].axhline(0, color='k', lw=0.6); ax[2].set_xlabel('додаткова яма α (ω², центр, r0 = 3)'); ax[2].set_ylabel('ω² найнижчої непарної моди')
ax[2].set_xlim(-0.005, 0.11); ax[2].set_ylim(-0.02, 0.03); ax[2].legend(fontsize=8); ax[2].set_title('в) ω² < 0: непарне поле «сідає» в ядро (поріг падає з розміром)', fontsize=9)
fig.suptitle('(306) Непарні кванти біля вихору Аліси бази: ядро сильно притягує, але безмасові фотон і «глюони» не дають зв\'язаного стану', fontsize=11)
plt.tight_layout(); plt.savefig('fig202_odd_core.png', dpi=130)
