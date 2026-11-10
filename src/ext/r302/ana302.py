"""(302) аналіз довгого зонда: (1) сила лінії ~0,54 і сусідніх смуг у ковзних вікнах 200 (непарні/парні); (2) точна частота; (3) кутове число ν на кільцях,
θ відлічується від напрямку розрізу Аліси (ланки з |U−1| > 1 біля ядра) — для півцілих ν розрив поля на розрізі тоді правильний."""
import numpy as np, json
D = '/mnt/project-files/model/s52/r302/'; zs = np.array([1, 1, -1, -1, 1.]); iu = np.triu_indices(5); odd = np.array([zs[i]*zs[j] < 0 for i, j in zip(*iu)])
k = np.load(D + 'probe302_kick.npz'); b = np.load(D + 'probe302_bg.npz'); a = k['rec'] - b['rec']; dts = float(k['dts']); tt = np.arange(len(a))*dts
dx, dy = k['dx'], k['dy']; r = np.hypot(dx, dy); cx, cy = float(k['cx']), float(k['cy'])
def detr(y):
    n = len(y); tl = np.arange(n); Y2 = y.reshape(n, -1); C = np.polynomial.polynomial.polyfit(tl, Y2, 2); return (Y2 - np.polynomial.polynomial.polyval(tl, C).T).reshape(y.shape)
def spec(y):
    n = len(y); h = np.hanning(n).reshape(-1, *([1]*(y.ndim - 1))); N = 16*n; F = np.abs(np.fft.fft(detr(y)*h, n=N, axis=0)[:N//2])**2
    return 2*np.pi*np.fft.fftfreq(N, dts)[:N//2], F
out = dict(windows=[])
core = r < 6
for t1 in range(0, 1000, 100):
    s = (tt >= t1) & (tt < t1 + 200)
    if s.sum() < 190: continue
    row = dict(t=t1 + 100)
    for lab, sel in (('odd', odd), ('even', ~odd)):
        om, F = spec(a[s][:, core][:, :, sel]); S = F.sum((1, 2))
        for nm, (lo, hi) in dict(b054=(0.50, 0.56), b029=(0.27, 0.33), b07=(0.68, 0.74), b08=(0.74, 0.90)).items(): row[f'{lab}_{nm}'] = float(S[(om > lo) & (om < hi)].sum())
        mm = (om > 0.45) & (om < 0.6); row[f'{lab}_peak'] = float(om[mm][np.argmax(S[mm])])
    out['windows'].append(row); print({kk: (round(v, 4) if 'peak' in kk else (f'{v:.2e}' if kk != 't' else v)) for kk, v in row.items() if kk in ('t', 'odd_b054', 'even_b054', 'odd_b07', 'odd_b08', 'odd_peak', 'even_peak')})
# частота й кутова структура в останніх 500
s = tt >= 500; y = detr(a[s]); n = len(y); tl = np.arange(n)
om, F = spec(a[s][:, core]); S = F.sum((1, 2)); mm = (om > 0.45) & (om < 0.6); om0 = float(om[mm][np.argmax(S[mm])]); print('om0 (t>500)', round(om0, 4))
ph = np.exp(-1j*om0*tl*dts)*np.hanning(n); Ap = np.einsum('t,tpc->pc', ph, y); Am = np.einsum('t,tpc->pc', np.conj(ph), y)
u = np.load(D + 'ucut_kick.npy'); L = 96
i, j = np.meshgrid(np.arange(L), np.arange(L), indexing='ij'); X = i + 0.5*j; Y = np.sqrt(3)/2*j
ang = []
for m, ev in enumerate([(1.0, 0.0), (0.5, np.sqrt(3)/2), (-0.5, np.sqrt(3)/2)]):
    mx, my = X + ev[0]/2 - cx, Y + ev[1]/2 - cy; sel = (u[m] > 1.0) & (np.hypot(mx, my) < 10) & (np.hypot(mx, my) > 1); ang += list(np.arctan2(my[sel], mx[sel]))
thc = float(np.angle(np.mean(np.exp(1j*np.array(ang))))) if ang else 0.0; print('cut links', len(ang), 'cut angle', round(np.degrees(thc), 1))
th = np.mod(np.arctan2(dy, dx) - thc, 2*np.pi)
nus = np.arange(-3, 3.01, 0.5); out['ang'] = {}
for ring in ((1.0, 3.0), (3.0, 5.5), (5.5, 9.0)):
    mr = (r >= ring[0]) & (r < ring[1])
    for lab, sel in (('odd', odd), ('even', ~odd), ('all', np.ones(15, bool))):
        pw = []
        for nu in nus:
            wv = np.exp(-1j*nu*th)*mr/mr.sum()
            pw.append(float(np.sum(np.abs(wv @ Ap[:, sel])**2) + np.sum(np.abs(wv @ Am[:, sel])**2)))
        pw = np.array(pw); pw /= pw.sum(); half = float(pw[np.abs(nus % 1 - 0.5) < 1e-9].sum())
        out['ang'][f'{ring}_{lab}'] = dict(nu=nus.tolist(), p=pw.tolist(), half=half)
        print(ring, lab, f'half {half:.2f}', ' '.join(f'{nu:+.1f}:{p:.2f}' for nu, p in zip(nus, pw) if p > 0.05))
pr = np.sum(np.abs(Ap)**2 + np.abs(Am)**2, 1); out['radial'] = dict(r=r.tolist(), p=(pr/pr.max()).tolist())
out['om0'] = om0; out['thc'] = thc
json.dump(out, open(D + 'ana302.json', 'w'))
