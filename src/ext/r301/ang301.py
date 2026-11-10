"""(301) кутова структура лінії ω≈0,54 у складеному вихорі (t 200–400): амплітуда на частоті в кожній точці → гармоніки e^{iνθ}."""
import numpy as np, json
D = '/mnt/project-files/model/s52/r301/'; zs = np.array([1, 1, -1, -1, 1.]); iu = np.triu_indices(5); odd = np.array([zs[i]*zs[j] < 0 for i, j in zip(*iu)])
k = np.load(D + 'probe_comp.npz'); b = np.load(D + 'probe_comp_bg.npz'); a = k['rec'] - b['rec']; dts = float(k['dts'])
dx, dy = k['dx'], k['dy']; r = np.hypot(dx, dy); th = np.arctan2(dy, dx); tt = np.arange(len(a))*dts
s = (tt >= 200) & (tt < 400); y = a[s]; n = len(y); tl = np.arange(n)
Y2 = y.reshape(n, -1); C = np.polynomial.polynomial.polyfit(tl, Y2, 2); Y2 = (Y2 - np.polynomial.polynomial.polyval(tl, C).T).reshape(y.shape)
om0 = 0.54; ph = np.exp(-1j*om0*tl*dts)*np.hanning(n)
Ap = np.einsum('t,tpc->pc', ph, Y2); Am = np.einsum('t,tpc->pc', np.conj(ph), Y2)   # положистa і від'ємна частоти
res = {}
for lab, sel in (('odd', odd), ('even', ~odd)):
    row = {}
    for nu in np.arange(-3, 3.01, 0.5):
        m = (r > 0.5) & (r < 6); wv = np.exp(-1j*nu*th)*m
        row[f'{nu:+.1f}'] = float(np.sum(np.abs(np.einsum('p,pc->c', wv, Ap[:, sel]))**2) + np.sum(np.abs(np.einsum('p,pc->c', wv, Am[:, sel]))**2))
    tot = sum(row.values()); res[lab] = {kk: v/tot for kk, v in row.items()}
    half = sum(v for kk, v in row.items() if abs(float(kk) % 1 - 0.5) < 1e-9)/tot
    print(lab, f'half-integer share {half:.2f}', ' '.join(f'{kk}:{v/tot:.2f}' for kk, v in row.items()))
    pw = np.sum(np.abs(Ap[:, sel])**2 + np.abs(Am[:, sel])**2, 1); print('   radial: r<2', pw[r < 2].sum()/pw.sum(), 'r2-4', pw[(r >= 2) & (r < 4)].sum()/pw.sum(), 'r4-7', pw[r >= 4].sum()/pw.sum())
json.dump(res, open(D + 'ang301.json', 'w'))
