"""(297) аналіз: маси вакууму за каналами (vac) і спектр ядра вихору за парністю щодо Z та кутовим числом ν (vort)."""
import json, numpy as np
D = '/mnt/project-files/model/s52/r297/'
odd = lambda i, j: (i in (2, 3)) != (j in (2, 3))
def spec(x, dts, w=None):
    tt_ = np.linspace(-1, 1, len(x)); V = np.vander(tt_, 4); cf = np.linalg.lstsq(V, x.reshape(len(x), -1), rcond=None)[0]
    x = x - (V @ cf).reshape(x.shape); h = np.hanning(len(x)); h = h.reshape((-1,) + (1,)*(x.ndim - 1))
    F = np.fft.fft(x*h, n=8*len(x), axis=0)[:4*len(x)]; om = 2*np.pi*np.fft.fftfreq(8*len(x), dts)[:4*len(x)]; return om, np.abs(F)**2
out = {}
v = np.load(D + 'probe_vac.npz'); rec = v['rec']; dts = float(v['dts']); iu, ia = v['iu'], v['ia']
names = [f'P{i}{j}' for i, j in zip(*iu)] + [f'E{m}_{i}{j}' for m in range(3) for i, j in zip(*ia)]
par = [odd(i, j) for i, j in zip(*iu)] + [odd(i, j) for m in range(3) for i, j in zip(*ia)]
om, S = spec(rec[20:], dts); vac = {}
for c, nm in enumerate(names):
    s = S[:, c]; k = np.argmax(s[om > 0.1]) + np.sum(om <= 0.1); vac[nm] = dict(peak=float(om[k]), odd=bool(par[c]), amp=float(s[k]))
out['vac'] = vac
for nm in names[:15] + names[15:25]: print('vac', nm, 'odd' if vac[nm]['odd'] else 'even', round(vac[nm]['peak'], 3), '%.2e' % vac[nm]['amp'])
z = np.load(D + 'probe_vort.npz'); rec = z['rec'].astype(np.float64) - np.load(D + 'probe_vort_bg.npz')['rec']; X, Y, cx, cy = z['X'], z['Y'], float(z['cx']), float(z['cy'])
th = np.mod(np.arctan2(Y - cy, X - cx) - np.pi/2, 2*np.pi); th = np.where((Y > cy) & (X < cx + 0.25) & (th > np.pi), th - 2*np.pi, th)
r = np.hypot(X - cx, Y - cy); w = ((r >= 1) & (r < 6)).astype(float)
nt = len(rec); tt = np.arange(nt)*dts; res = {}
for parity in (False, True):
    cs = [c for c in range(45) if par[c] == parity]
    nus = np.arange(-2.5, 3, 1.0) if parity else np.arange(-3, 4, 1.0)
    for nu in nus:
        a = np.einsum('tnc,n->tc', rec[:, :, cs], w*np.exp(-1j*nu*th))
        for win, (t1, t2) in dict(early=(0, 100), late=(150, 400)).items():
            sel = (tt >= t1) & (tt < t2); om, S = spec(a[sel], dts); St = S.sum(1)
            res[f'{"odd" if parity else "even"}_{nu:g}_{win}'] = dict(om=om[om < 2.5].tolist(), S=St[om < 2.5].tolist())
out['vort'] = res
json.dump(out, open(D + 'ana297.json', 'w'))
def band(q, lo, hi): om, S = np.array(q['om']), np.array(q['S']); m = (om > lo) & (om < hi); return S[m].sum()*(om[1] - om[0])
for k, q in res.items():
    if not k.endswith('late'): continue
    qe = res[k.replace('late', 'early')]; om, S = np.array(q['om']), np.array(q['S']); m = om > 0.15
    pk = []
    for i in np.argsort(np.where(m, S, 0))[::-1]:
        if all(abs(om[i] - p) > 0.04 for p in pk): pk.append(round(float(om[i]), 3))
        if len(pk) == 4: break
    bands = [(0.15, 0.40), (0.40, 0.47), (0.47, 1.0), (1.0, 2.5)]
    rat = [band(q, *bd)/250/max(band(qe, *bd)/100, 1e-30) for bd in bands]
    print(k[:-5], 'peaks', pk, 'late/early по смугах', ' '.join('%.3f' % x for x in rat))
