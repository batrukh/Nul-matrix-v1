import json, numpy as np
from scipy.optimize import least_squares
d = json.load(open('ana292.json'))
def sim(F, m0, c, t, kind):
    v = 0.0; p = 0.0; x = 0.0; out = []; dt = 0.01; tt = 0.0; k = 0
    for T in t:
        while tt < T - 1e-9:
            gam = 1/np.sqrt(1 - min(v/c, 0.9999)**2)
            p += (F if kind == 'A' else F/gam)*dt
            v = p/np.sqrt(m0**2 + (p/c)**2); x += v*dt; tt += dt
        out.append(x)
    return np.array(out)
def run(kind, gs, tmax):
    def res(q):
        m0, c, t0 = q; r = []
        for g in gs:
            t = np.array(d[g]['t']); s = t <= tmax[g]; x = np.array(d[g]['x'])[s]
            r.append(sim(d[g]['F'], m0, c, t[s] + t0, kind) - sim(d[g]['F'], m0, c, [t0], kind)[0] - x)
        return np.concatenate(r)
    f = least_squares(res, [1.35, 0.8, 2.0], bounds=([0.5, 0.3, 0], [5, 3, 10]))
    return f.x, np.sqrt(np.mean(f.fun**2))
tm = {'0.01': 80, '0.02': 80, '0.04': 40}
for kind in 'AB':
    print(kind, run(kind, ['0.01', '0.02', '0.04'], tm))
    for g in ['0.01', '0.02', '0.04']: print('  ', g, run(kind, [g], tm))
# Ньютон (c→∞)
def newt(gs):
    def res(q):
        m0, t0 = q; return np.concatenate([np.array(d[g]['x'])[np.array(d[g]['t']) <= tm[g]] - d[g]['F']/(2*m0)*((np.array(d[g]['t'])[np.array(d[g]['t']) <= tm[g]] + t0)**2 - t0**2) for g in gs])
    f = least_squares(res, [1.35, 1.0]); return f.x, np.sqrt(np.mean(f.fun**2))
print('Newton', newt(['0.01', '0.02', '0.04']))
