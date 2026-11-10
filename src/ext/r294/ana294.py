import json, numpy as np
from scipy.optimize import least_squares
L = 96; out = {}
def sim(F, m0, c, t, kind):
    v = p = x = 0.0; o = []; dt = 0.01; tt = 0.0
    for T in t:
        while tt < T - 1e-9:
            gam = 1/np.sqrt(1 - min(v/c, 0.9999)**2); p += (F if kind == 'A' else F/gam)*dt; v = p/np.sqrt(m0**2 + (p/c)**2); x += v*dt; tt += dt
        o.append(x)
    return np.array(o)
D = {}
for g in ['0.01', '0.02', '0.04']:
    d = json.load(open(f'mass_ka0.5_g{g}.json')); r = d['rec']; F = 0.921*float(g)
    t = np.array([q['t'] for q in r]); x = np.array([[q['p'][i][0] for i in range(2)] for q in r])
    x = np.unwrap(x*2*np.pi/L, axis=0)*L/(2*np.pi); x = x.mean(1) - x[0].mean(); v = np.gradient(x, t)
    wr = np.array([np.mean([w[0] for w in q['w']])/np.mean([w[1] for w in q['w']]) for q in r])
    D[g] = dict(t=t, x=x, v=v, F=F, wr=wr/wr[0])
    for tt in [10, 20, 40, 60, 80]:
        i = np.argmin(abs(t - tt)); print(g, tt, 'x %.2f v %.3f wr %.3f' % (x[i], v[i], wr[i]/wr[0]))
tm = {'0.01': 80, '0.02': 80, '0.04': 80}
def run(kind, gs):
    def res(q):
        m0, c = q; o = []
        for g in gs:
            t = D[g]['t']; s = t <= tm[g]; o.append(sim(D[g]['F'], m0, c, t[s] + 2, kind) - sim(D[g]['F'], m0, c, [2], kind)[0] - D[g]['x'][s])
        return np.concatenate(o)
    f = least_squares(res, [1.0, 1.0], bounds=([0.3, 0.3], [5, 3])); return f.x, float(np.sqrt(np.mean(f.fun**2)))
for kind in 'AB':
    r = run(kind, ['0.01', '0.02', '0.04']); print(kind, r); out[kind] = [r[0].tolist(), r[1]]
    for g in ['0.01', '0.02', '0.04']: rr = run(kind, [g]); print('  ', g, rr); out[kind + g] = [rr[0].tolist(), rr[1]]
# small-v inertial mass: early quadratic
for g in ['0.01', '0.02']:
    t, x = D[g]['t'], D[g]['x']; s = (t > 3) & (t < 25); p = np.polyfit(t[s], x[s], 2); print('m_I early', g, D[g]['F']/(2*p[0]))
json.dump(dict(out=out, D={g: {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in D[g].items()} for g in D}), open('ana294.json', 'w'))
