"""(285) аналіз: знакова кореляція C = Σφ_v·φ_c / Σφ_c² і відношення флюенсів за вихором; порівняння з АБ α=½ (abtheory.ab_sim)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, json, glob, abtheory as A
res = {}
for lam in (6, 8, 12):
    fv, fc = f'abw_vort_L128_lam{lam}_c0.65.npz', f'abw_ctrl_L128_lam{lam}_c0.65.npz'
    try: V, Cc = np.load(fv), np.load(fc)
    except FileNotFoundError: continue
    X, Y = V['X'], V['Y']; cx, cy = 80.0, float(Y[0, 64])
    sv, sc = V['sig'].astype(float), Cc['sig'].astype(float)
    num = (sv*sc).sum(0); den = (sc**2).sum(0); C = num/np.maximum(den, 1e-30)
    R = V['fl']/np.maximum(Cc['fl'], 1e-30)
    out = {}
    for d in (8, 12, 16, 20, 24):
        row = []
        for dy in np.arange(-16, 16.5, 1.0):
            r = np.hypot(X - (cx + d), Y - (cy + dy)); k = np.unravel_index(np.argmin(r), r.shape)
            if r[k] > 0.8: continue
            th = A.ab_sim(np.array([float(X[k] - cx)]), np.array([float(Y[k] - cy)]), 2*np.pi/lam)[0]/np.exp(1j*2*np.pi/lam*(X[k] - cx))
            row.append(dict(dy=float(Y[k] - cy), C=float(C[k]), R=float(R[k]), ab_re=float(th.real), ab_I=float(abs(th)**2), wc=float(den[k])))
        out[d] = row
    # фронтальна область (до вихора) — контроль методу
    pre = (X > cx - 20) & (X < cx - 8) & (np.abs(Y - cy) < 10)
    res[lam] = dict(rows=out, C_pre=float(np.median(C[pre])), R_pre=float(np.median(R[pre])))
    print('λ', lam, 'перед вихором: C %.3f, R %.3f' % (res[lam]['C_pre'], res[lam]['R_pre']))
    for d in (12, 20):
        rr = out[d]; up = [q for q in rr if q['dy'] > 4]; dn = [q for q in rr if q['dy'] < -4]; ax = [q for q in rr if abs(q['dy']) < 1.5]
        f = lambda L_, key: np.mean([q[key] for q in L_]) if L_ else np.nan
        print('  d', d, 'C вище %.2f (АБ %.2f), нижче %.2f (АБ %.2f), на осі %.2f; I на осі %.2f (АБ %.2f)' %
              (f(up, 'C'), f(up, 'ab_re'), f(dn, 'C'), f(dn, 'ab_re'), f(ax, 'C'), f(ax, 'R'), f(ax, 'ab_I')))
json.dump(res, open('ana285.json', 'w'))
