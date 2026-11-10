"""(307) рис. 203: маса непарних глюонів — зв'язаний заряджений непарний стан у ядрі вихору"""
import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
def low(f):
    for q in json.load(open(f))['modes']:
        if q['odd'] > 0.5 and q['q2'] > 0.5 and q['w2'] > 1e-4: return q['w'], q['core']
    print('none', f); return np.nan, np.nan
MG = [0.15, 0.2, 0.3, 0.4, 0.5]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
a = ax[0]; a.plot([0.1, 0.55], [0.1, 0.55], 'k:', lw=1, label='ω = m (поріг глюона)')
for f, lab, st in [('V20/m2_vac_rc0.0_mg{}.json', 'без вихору, маса всюди (поріг у r<20)', 's-'), ('V20/m2_vac_mg{}.json', 'без вихору, маса з діркою r<3,5', 'D--'),
                   ('R20/m2_v_mg{}.json', 'вихор, дірка r<3,5', 'o-'), ('R20/m2_v_rc2.5_mg{}.json', 'вихор, дірка r<2,5', 'o--'), ('R20/m2_v_rc2.0_mg{}.json', 'вихор, дірка r<2', 'o:')]:
    a.plot(MG, [low(f.format(m))[0] for m in MG], st, ms=5, label=lab)
a.set_xlabel('маса непарних глюонів m'); a.set_ylabel('найнижча заряджена непарна ω'); a.set_title('а) r<20: вихор тримає квант нижче порогу'); a.legend(fontsize=8); a.grid(alpha=.3)
m = np.load('R20/meta.npz'); d = np.load('R20/m2vec_mg0.3.npz'); js = json.load(open('R20/m2_v_mg0.3.json'))['modes']
k = next(i for i, q in enumerate(js) if q['odd'] > 0.5 and q['q2'] > 0.5)
mass, sites, NP, aidx = m['mass'], m['sites'], int(m['NP']), m['aidx']
x = d['V'][:, k]/np.sqrt(mass); rho = np.zeros(m['rw'].shape)
rho[sites[:, 0], sites[:, 1]] += (x[:14*NP]**2).reshape(NP, 14).sum(1)
for mm in range(3):
    for (i, j) in np.argwhere(aidx[mm] >= 0): rho[i, j] += np.sum(x[14*NP + 10*aidx[mm, i, j] + np.arange(10)]**2)
X = sites[:, 0] + 0.5*sites[:, 1]; Y = np.sqrt(3)/2*sites[:, 1]; c = rho[sites[:, 0], sites[:, 1]]
b = ax[1]; sc = b.scatter(X, Y, c=c/c.max(), s=14, cmap='magma'); b.set_aspect('equal'); plt.colorbar(sc, ax=b, shrink=.8)
b.set_title(f'б) густина зв\'язаного стану, m=0,3: ω={js[k]["w"]:.3f}, у ядрі {js[k]["core"]*100:.0f} %'); b.set_xticks([]); b.set_yticks([])
plt.tight_layout(); plt.savefig('fig203_bound_odd_gluon.png', dpi=130)
for f in ['R20/m2_v_mg{}.json', 'R20/m2_v_rc2.5_mg{}.json', 'R20/m2_v_rc2.0_mg{}.json', 'V20/m2_vac_mg{}.json', 'V20/m2_vac_rc0.0_mg{}.json']:
    print(f, [tuple(round(v, 3) for v in low(f.format(mm))) for mm in MG])
