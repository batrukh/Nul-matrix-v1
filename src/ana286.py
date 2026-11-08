"""(286) аналіз: C = Σ⟨δE_v,δE_c⟩/Σ⟨δE_c,δE_c⟩ і відношення енергій за вихором уздовж і поперек пучка; розріз іде в бік +pe,
тож для струни Аліси очікуємо C ≈ −1 з боку +pe і +1 з боку −pe за вихором, вузол на осі; перед вихором C ≈ 1."""
import os as _o; _V = _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), 'variant'); RUNS = _o.environ.get('NM_RUNS', _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), '..', 'runs', 'out', 'variant'))
import sys, json, glob, numpy as np
sys.path.insert(0, _V); import matter98 as M98, phi90 as PH
sys.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); import abtheory as A
L = 96; res = {}
for f in sorted(glob.glob('rw_s*_lam*.json')):
    J = json.load(open(f)); Z = np.load(f[:-5] + '.npz'); X, Y = (Z['X'], Z['Y']) if 'X' in Z else PH.coords(L)
    C = Z['num']/np.maximum(Z['den'], 1e-30); R = Z['flv']/np.maximum(Z['den'], 1e-30); dist = Z['dist']; den = Z['den']
    vx, vy = J['v']; ax = np.array(J['ax']); pe = np.array(J['pe']); k = 2*np.pi/J['lam']
    rows = []
    for d in (-6, 4, 6, 8, 10):
        for yy in np.arange(-10, 10.5, 1.0):
            qx, qy = vx + d*ax[0] + yy*pe[0], vy + d*ax[1] + yy*pe[1]
            rr, _, _ = M98.wrapd(X, Y, qx, qy, L); n = np.unravel_index(np.argmin(rr), rr.shape)
            th = A.ab_sim(np.array([float(d)]), np.array([float(yy)]), k)[0]/np.exp(1j*k*d)
            rows.append(dict(d=d, y=float(yy), C=float(C[n]), R=float(R[n]), dist=float(dist[n]), w=float(den[n]), ab=float(th.real), abI=float(abs(th)**2)))
    sel = lambda dd, lo, hi: [q for q in rows if q['d'] in dd and lo <= q['y'] <= hi and q['dist'] >= 1.5 and q['w'] > 0.05*den.max()]
    wm = lambda L_, key: float(np.average([q[key] for q in L_], weights=[q['w'] for q in L_])) if L_ else float('nan')
    S = dict(up=wm(sel((4, 6, 8, 10), 2.5, 10), 'C'), dn=wm(sel((4, 6, 8, 10), -10, -2.5), 'C'), axis_C=wm(sel((4, 6, 8, 10), -1, 1), 'C'),
             axis_R=wm(sel((4, 6, 8, 10), -1, 1), 'R'), pre=wm(sel((-6,), -6, 6), 'C'),
             ab_up=wm(sel((4, 6, 8, 10), 2.5, 10), 'ab'), ab_dn=wm(sel((4, 6, 8, 10), -10, -2.5), 'ab'), ab_axisI=wm(sel((4, 6, 8, 10), -1, 1), 'abI'))
    res[f[:-5]] = dict(json=J, rows=rows, summary=S)
    print(f[:-5], 'глиб. %.0f' % J['depth'], 'hol', J['hol']['vort_R3'][0], J['hol']['ctrl_R3'][0],
          '| перед C %.2f | за: C(+pe) %.2f [АБ %.2f]  C(−pe) %.2f [АБ %.2f]  вісь C %.2f, енергія %.2f [АБ %.2f]' %
          (S['pre'], S['up'], S['ab_up'], S['dn'], S['ab_dn'], S['axis_C'], S['axis_R'], S['ab_axisI']))
json.dump(res, open('ana286.json', 'w'))
