import json, numpy as np
m0 = 0.921/0.68; L = 96
out = {}
for g in ['0.01', '0.02', '0.04']:
    d = json.load(open(f'mass_g{g}.json')); r = d['rec']; F = 0.921*float(g)
    t = np.array([q['t'] for q in r]); x = np.array([[q['p'][i][0] for i in range(2)] for q in r])
    x = np.unwrap(x*2*np.pi/L, axis=0)*L/(2*np.pi); x = x.mean(1) - x[0].mean()
    v = np.gradient(x, t); wx = np.array([np.mean([w[0] for w in q['w']]) for q in r]); wy = np.array([np.mean([w[1] for w in q['w']]) for q in r])
    E = np.array([q['E'] for q in r])
    out[g] = dict(t=t.tolist(), x=x.tolist(), v=v.tolist(), wx=wx.tolist(), wy=wy.tolist(), E=E.tolist(), F=F)
    for tt in [5, 10, 20, 30, 40, 60, 80]:
        i = np.argmin(abs(t - tt)); p = F*t[i]
        print(g, tt, f'x {x[i]:.2f} v {v[i]:.3f} m=p/v {p/max(v[i],1e-9):.2f} wx/wy {wx[i]/wy[i]:.3f} dE {(E[i]-E[0])/2:.3f} work {F*x[i]:.3f}')
json.dump(out, open('ana292.json', 'w'))
