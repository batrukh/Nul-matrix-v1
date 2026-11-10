"""(293) підгонка двох затухаючих коливань (дихання ≈0,38 і радіальна мода ≈1,4) до різниці «з поштовхом − без»."""
import json, numpy as np
from scipy.optimize import curve_fit
def f(t, A, w, ph, gm, B, W, PH, GM, c0, c1):
    return A*np.exp(-gm*t)*np.cos(w*t + ph) + B*np.exp(-GM*t)*np.cos(W*t + PH) + c0 + c1*t
lo = [-1, 0.2, -10, -0.05, -1, 1.0, -10, -0.05, -1, -1]; hi = [1, 0.55, 10, 1, 1, 1.6, 10, 1, 1, 1]
def fit(t, s):
    best = None
    for wg in np.linspace(0.28, 0.45, 9):
        for ph in [0, 1.5, 3, 4.5]:
            try:
                p, cov = curve_fit(f, t - t[0], s, p0=[np.std(s), wg, ph, 0.01, np.std(s), 1.35, 0, 0.02, 0, 0], bounds=(lo, hi), maxfev=20000)
                r = np.sqrt(np.mean((f(t - t[0], *p) - s)**2))
                if best is None or r < best[2]: best = (p, np.sqrt(np.diag(cov)), r)
            except Exception: pass
    return best
out = {}
r = json.load(open('mv_g0_acc0.json'))['rec']; t = np.array([q['t'] for q in r])
for span in [40, 70]:
    s = np.array([q['s'][0][0] for q in r]); m = (t > 5) & (t < 5 + span); p, e, res = fit(t[m], s[m])
    print('rest', span, 'breath w=%.4f±%.4f gam=%.4f A=%.5f | radial W=%.4f±%.4f' % (p[1], e[1], p[3], p[0], p[5], e[5]), 'rel res %.2f' % (res/np.std(s[m])))
    out[f'rest_{span}'] = dict(w=p[1], ew=e[1], gam=p[3], A=p[0], W=p[5], eW=e[5], GM=p[7])
for g in ['0.02', '0.04']:
    a = json.load(open(f'kk_g{g}_k0.1.json'))['rec']; b = json.load(open(f'kk_g{g}_k0.json'))['rec']; n = min(len(a), len(b)); t = np.array([q['t'] for q in a[:n]])
    for span in [40, 70]:
        s = np.array([q['s'][0][0] for q in a[:n]]) - np.array([q['s'][0][0] for q in b[:n]]); m = (t > 41) & (t < 41 + span)
        p, e, res = fit(t[m], s[m])
        print(g, span, 'breath w=%.4f±%.4f gam=%.4f A=%.5f | radial W=%.4f±%.4f GM=%.3f' % (p[1], e[1], p[3], p[0], p[5], e[5], p[7]), 'rel res %.2f' % (res/np.std(s[m])))
        out[f'{g}_{span}'] = dict(w=p[1], ew=e[1], gam=p[3], A=p[0], W=p[5], eW=e[5], GM=p[7])
json.dump(out, open('fit293.json', 'w'))
