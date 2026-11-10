"""(301) сила смуги ω (поштовх − фон) у ковзних вікнах: чи живе лінія 0,535 у складеному вихорі."""
import numpy as np, json
D = '/mnt/project-files/model/s52/r301/'; zs = np.array([1, 1, -1, -1, 1.]); iu = np.triu_indices(5); odd = np.array([zs[i]*zs[j] < 0 for i, j in zip(*iu)])
res = {}
for mode in ('comp', 'alice'):
    k = np.load(D + f'probe_{mode}.npz'); b = np.load(D + f'probe_{mode}_bg.npz'); a = k['rec'] - b['rec']; dts = float(k['dts'])
    r = np.hypot(k['dx'], k['dy']); w = r < 6; tt = np.arange(len(a))*dts
    for lab, sel in (('odd', odd), ('even', ~odd)):
        x = a[:, w][:, :, sel]; rows = []
        for t1 in range(0, 400, 50):
            s = (tt >= t1) & (tt < t1 + 100)
            if s.sum() < 150: continue
            y = x[s]; n = len(y); tl = np.arange(n)
            y = y - np.polyval(np.polyfit(tl, y.reshape(n, -1), 2), tl[:, None]*0 + tl[:, None]) if False else y
            # прибрати повільний дрейф: квадратична підгонка по часу для кожного каналу
            Y2 = y.reshape(n, -1); C = np.polynomial.polynomial.polyfit(tl, Y2, 2); Y2 = Y2 - np.polynomial.polynomial.polyval(tl, C).T
            h = np.hanning(n)[:, None]; N = 16*n; F = np.abs(np.fft.fft(Y2*h, n=N, axis=0)[:N//2])**2; om = 2*np.pi*np.fft.fftfreq(N, dts)[:N//2]
            S = F.sum(1); band = lambda lo, hi: float(S[(om > lo) & (om < hi)].sum())
            rows.append(dict(t=t1 + 50, b053=band(0.50, 0.55), b029=band(0.27, 0.32), b07=band(0.69, 0.74), b076=band(0.74, 0.80), peak=float(om[(om > 0.2) & (om < 1.6)][np.argmax(S[(om > 0.2) & (om < 1.6)])])))
        res[f'{mode}_{lab}'] = rows
        print(mode, lab); [print('  ', {kk: (f'{v:.2e}' if isinstance(v, float) and kk != 'peak' and kk != 't' else v) for kk, v in rr.items()}) for rr in rows]
json.dump(res, open(D + 'band301.json', 'w'))
