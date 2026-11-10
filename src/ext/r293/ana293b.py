import json, numpy as np
L = 96; out = {}
def spec(t, s, lo=0.2, hi=0.7):
    x = (s - s.mean())*np.hanning(len(s)); dt = t[1]-t[0]
    F = np.abs(np.fft.rfft(x, 64*len(x))); w = 2*np.pi*np.fft.rfftfreq(64*len(x), dt); sel = (w > lo) & (w < hi); return w[sel], F[sel]/F[sel].max()
# спокій: br_br_0.1 (той самий сигнал? там середнє r<2) — беремо як ω0
for gname in ['0.02', '0.04']:
    a = json.load(open(f'kk_g{gname}_k0.1.json'))['rec']; b = json.load(open(f'kk_g{gname}_k0.json'))['rec']
    n = min(len(a), len(b)); t = np.array([q['t'] for q in a[:n]])
    for vi in range(2):
        x = np.array([q['p'][vi][0] for q in b[:n]]); x = np.unwrap(x*2*np.pi/L)*L/2/np.pi
        m = t >= 41; v = float(np.polyfit(t[m], x[m], 1)[0])
        res = {}
        for k, nm in [(0, 'trP2'), (1, 'E3')]:
            s = np.array([q['s'][vi][k] for q in a[:n]]) - np.array([q['s'][vi][k] for q in b[:n]])
            w, F = spec(t[m], s[m]); i = np.argmax(F)
            # amplitude decay
            env = [float(np.std(s[(t >= lo) & (t < lo + 30)])) for lo in (41, 71, 101, 131)]
            res[nm] = dict(w=float(w[i]), env=env, spec=(w[::16].tolist(), F[::16].tolist()))
        out[f'{gname}_{vi}'] = dict(v=v, **res)
        print(gname, vi, 'v=%.3f' % v, {nm: (round(res[nm]['w'], 4), [round(e, 6) for e in res[nm]['env']]) for nm in ['trP2', 'E3']})
json.dump(out, open('ana293b.json', 'w'))
w0 = 0.385
for v in [0.35, 0.49]:
    for c in [0.70, 0.77, 1.0]: print(v, c, round(w0*np.sqrt(1 - (v/c)**2), 4))
