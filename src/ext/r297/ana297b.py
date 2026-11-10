"""(297) аналіз поштовху лише P (закон Гаусса): спектр непарного відгуку в ядрі за кутовими числами ν, вікна 0–100 і 200–800."""
import json, sys, numpy as np
D = '/mnt/project-files/model/s52/r297/'; TAG = sys.argv[1] if len(sys.argv) > 1 else 'oddP800'
a = np.load(D + f'diff_{TAG}.npy'); z = np.load(D + 'probe_vort_bg_800.npz'); X, Y, cx, cy, dts = z['X'], z['Y'], float(z['cx']), float(z['cy']), float(z['dts'])
zs = np.array([1, 1, -1, -1, 1.]); iu = np.triu_indices(5); ia = np.triu_indices(5, 1)
nm = [f'P{i}{j}' for i, j in zip(*iu)] + [f'E{m}_{i}{j}' for m in range(3) for i, j in zip(*ia)]
odd = np.array([zs[i]*zs[j] < 0 for i, j in zip(*iu)] + [zs[i]*zs[j] < 0 for m in range(3) for i, j in zip(*ia)])
th = np.mod(np.arctan2(Y - cy, X - cx) - np.pi/2, 2*np.pi); th = np.where((Y > cy) & (X < cx + 0.25) & (th > np.pi), th - 2*np.pi, th)
r = np.hypot(X - cx, Y - cy); tt = np.arange(len(a))*dts
def spec(x):
    x = x - x.mean(0); h = np.hanning(len(x))[:, None]; n = 16*len(x)
    F = np.fft.fft(x*h, n=n, axis=0)[:n//2]; return 2*np.pi*np.fft.fftfreq(n, dts)[:n//2], np.abs(F)**2
out = {}
for nu in np.arange(-2.5, 3, 1.0):
    w = ((r >= 0.5) & (r < 6))*np.exp(-1j*nu*th)
    for lab, (t1, t2) in dict(early=(0, 100), late=(200, 800)).items():
        s = (tt >= t1) & (tt < t2); A = np.einsum('tnc,n->tc', a[s][:, :, odd], w)
        om, S = spec(A); m = om < 2.0
        out[f'{nu:g}_{lab}'] = dict(om=om[m].tolist(), S=S[m].sum(1).tolist(), Sc=S[m].tolist())
names_odd = [n for n, o in zip(nm, odd) if o]
for nu in np.arange(-2.5, 3, 1.0):
    q = out[f'{nu:g}_late']; om, S, Sc = np.array(q['om']), np.array(q['S']), np.array(q['Sc'])
    pk = []
    for i in np.argsort(S)[::-1]:
        if all(abs(om[i] - p) > 0.03 for p, _ in pk): pk.append((round(float(om[i]), 3), float(S[i])))
        if len(pk) == 4: break
    tot = S.sum()
    desc = []
    for p, sv in pk:
        i = np.argmin(abs(om - p)); top = np.argsort(Sc[i])[::-1][:3]
        desc.append(f'{p} ({sv/S.max():.2f}; ' + ','.join(names_odd[j] for j in top) + ')')
    print(f'nu {nu:+.1f} late power {tot:.2e} early {np.sum(out[f"{nu:g}_early"]["S"]):.2e} |', ' '.join(desc))
json.dump(dict(tag=TAG, nm=names_odd, spec={k: dict(om=v['om'][::4], S=v['S'][::4]) for k, v in out.items()}), open(D + f'ana297b_{TAG}.json', 'w'))
