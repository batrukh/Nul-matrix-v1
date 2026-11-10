"""(308)–(309) рис. 204"""
import numpy as np, json, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
E = {k: float(np.load(f'R20/dir_{o}_k{k}.npz')['E']) for k in ['0.09'] for o in ['e0']}
pairs = [('k 0,09, κ 1', float(np.load('R20/dir_e2_k0.09.npz')['E']) - float(np.load('R20/dir_e0_k0.09.npz')['E'])),
         ('k 0,04, κ 0,3', float(np.load('R20/dir_e2_k0.04.npz')['E']) - float(np.load('R20/dir_e0_k0.04.npz')['E']))]
a = ax[0]; a.bar([p[0] for p in pairs], [p[1] for p in pairs], color='#c0504d'); a.set_ylabel('E(старт e2) − E(старт e0)')
a.set_title('а) директор біля вихору: старт уздовж e2\nсповзає до парної осі, енергія вища'); a.grid(alpha=.3, axis='y')
def low(f, cond):
    for q in json.load(open(f))['modes']:
        if q['odd'] > 0.5 and cond(q) and q['w2'] > 1e-4: return q['w']
b = ax[1]
vals = [('руками e2\n(307)', low('../r306/R20/m2_v_mg0.3.json', lambda q: q['q2'] > 0.5)), ('директор e2\nвакуум', low('V20/m3_dir_e2_k0.09.json', lambda q: q['q2'] > 0.5)),
        ('директор\nрелаксований', low('R20/m3_dir_e0_k0.09.json', lambda q: True)), ('без маси\n(306)', 0.1113)]
b.bar([v[0] for v in vals], [v[1] for v in vals], color=['#4f81bd', '#999999', '#c0504d', '#f79646']); b.axhline(0.3, ls=':', c='k'); b.text(3.4, 0.305, 'm', fontsize=9)
b.set_ylabel('найнижча непарна ω, r<20'); b.set_title('б) з природним директором непарний\nG12 безмасовий — кванта не тримає'); b.grid(alpha=.3, axis='y')
# в) фаза u+ на кільці
sys.argv = ['', '../r306/R20/m2vec_mg0.3.npz', '../r306/R20/m2_v_mg0.3.json', '/dev/null']
exec(open('nu2.py').read().split('out = []')[0])
k = next(i for i, q in enumerate(js) if q['odd'] > 0.5 and q['q2'] > 0.5)
x = d['V'][:, k]/np.sqrt(mass); P_, U_ = [], []
for (a_, b_) in np.argwhere((rw > 5) & (rw < 7.5)):
    Fm = F.get((a_, b_))
    if Fm is None or any(aidx[mm, a_, b_] < 0 for mm in range(3)): continue
    c = [np.sum((Fm.T @ np.einsum('k,kij->ij', x[14*NP + 10*aidx[mm, a_, b_] + np.arange(10)], T) @ Fm)*G02) + 1j*np.sum((Fm.T @ np.einsum('k,kij->ij', x[14*NP + 10*aidx[mm, a_, b_] + np.arange(10)], T) @ Fm)*G12) for mm in range(3)]
    V = (2/3)*np.einsum('m,mi->i', np.array(c), dh); P_.append(np.arctan2(np.sqrt(3)/2*b_ - vy, a_ + 0.5*b_ - vx)); U_.append(V[0] + 1j*V[1])
P_, U_ = np.array(P_), np.array(U_); o = np.argsort(P_)
cc = ax[2]; cc.plot(P_[o], U_[o].real/abs(U_).max(), '.', label='Re u₊'); cc.plot(P_[o], U_[o].imag/abs(U_).max(), '.', label='Im u₊')
cc.axvline(np.pi, c='k', lw=.8); cc.axvline(-np.pi, c='k', lw=.8); cc.set_xlabel('кут φ довкола вихору (розріз при ±π)'); cc.legend(fontsize=8)
cc.set_title('в) зв\'язаний квант (307), кільце r 5–7,5:\nна розрізі знак змінюється — ν півцілий'); cc.grid(alpha=.3)
plt.tight_layout(); plt.savefig('fig204_holder_spin.png', dpi=130)
