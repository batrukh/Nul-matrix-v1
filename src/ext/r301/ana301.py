"""(301) аналіз: спектр відгуку в ядрі (поштовх − фон), непарні й парні компоненти Φ, вікна 0–100 і 100–400; вакуум — k=0."""
import json, numpy as np
D = '/mnt/project-files/model/s52/r301/'
zs = np.array([1, 1, -1, -1, 1.]); iu = np.triu_indices(5); odd = np.array([zs[i]*zs[j] < 0 for i, j in zip(*iu)])
def spec(x, dts):
    x = x - x.mean(0); h = np.hanning(len(x)).reshape(-1, *([1]*(x.ndim - 1))); n = 16*len(x)
    F = np.fft.fft(x*h, n=n, axis=0)[:n//2]; return 2*np.pi*np.fft.fftfreq(n, dts)[:n//2], np.abs(F)**2
def peaks(om, S, k=5, lo=0.03, hi=1.6):
    m = (om > lo) & (om < hi); o, s = om[m], S[m]; pk = []
    for i in np.argsort(s)[::-1]:
        if all(abs(o[i] - p) > 0.04 for p, _ in pk): pk.append((round(float(o[i]), 3), float(s[i]/s.max())))
        if len(pk) == k: break
    return pk
out = {}
v = np.load(D + 'probe_vac.npz'); a = v['rec']; dts = float(v['dts'])
for lab, sel in (('odd', odd), ('even', ~odd)):
    om, S = spec(a[:, sel], dts); S = S.sum(1); out[f'vac_{lab}'] = dict(om=om[om < 2].tolist(), S=S[om < 2].tolist()); print('vac', lab, peaks(om, S))
for mode in ('comp', 'alice'):
    k = np.load(D + f'probe_{mode}.npz'); b = np.load(D + f'probe_{mode}_bg.npz'); a = k['rec'] - b['rec']; dts = float(k['dts'])
    r = np.hypot(k['dx'], k['dy']); w = (r < 6)
    tt = np.arange(len(a))*dts
    for lab, sel in (('odd', odd), ('even', ~odd)):
        for win, (t1, t2) in dict(early=(0, 100), late=(100, 400)).items():
            s = (tt >= t1) & (tt < t2); om, S = spec(a[s][:, w][:, :, sel], dts); S = S.sum((1, 2))
            out[f'{mode}_{lab}_{win}'] = dict(om=om[om < 2].tolist(), S=S[om < 2].tolist())
            print(mode, lab, win, f'power {S.sum():.2e}', peaks(om, S))
    # загасання: енергія непарного відгуку в ядрі з часом
    e = (np.abs(a[:, w][:, :, odd])**2).sum((1, 2)); out[f'{mode}_odd_decay'] = dict(t=tt[::10].tolist(), e=e[::10].tolist())
    e2 = (np.abs(a[:, w][:, :, ~odd])**2).sum((1, 2)); out[f'{mode}_even_decay'] = dict(t=tt[::10].tolist(), e=e2[::10].tolist())
    print(mode, 'odd energy t~5/100/200/400:', [f'{e[np.argmin(abs(tt - x))]:.2e}' for x in (5, 100, 200, 399)])
json.dump(out, open(D + 'ana301.json', 'w'))
