import json, numpy as np
L = 96; out = {}
def spec(t, s, lo=0.15, hi=0.7):
    x = s - np.polyval(np.polyfit(t, s, 2), t); x = x*np.hanning(len(x)); dt = t[1] - t[0]
    F = np.abs(np.fft.rfft(x, 32*len(x))); w = 2*np.pi*np.fft.rfftfreq(32*len(x), dt); sel = (w > lo) & (w < hi)
    return w[sel], F[sel]
def peak(w, F):
    i = np.argmax(F); return float(w[i])
for f, tacc in [('mv_g0_acc0.json', 0), ('mv_g0.02_acc30.json', 30), ('mv_g0.04_acc25.json', 25)]:
    d = json.load(open(f)); r = d['rec']; t = np.array([q['t'] for q in r])
    x = np.array([np.mean([p[0] for p in q['p']]) for q in r]); x = np.unwrap(x*2*np.pi/L)*L/2/np.pi
    m = t >= tacc + 10; v = float(np.polyfit(t[m], x[m], 1)[0])
    wr = np.array([np.mean([w[0] for w in q['w']])/np.mean([w[1] for w in q['w']]) for q in r])
    res = dict(v=v, wr=float(wr[m].mean()), wr0=float(wr[0]))
    for k, nm in [(0, 'trP2'), (1, 'E3')]:
        ws = []; 
        for vi in range(2):
            s = np.array([q['s'][vi][k] for q in r]); w, F = spec(t[m], s[m]); ws.append(peak(w, F)); res[f'{nm}_{vi}_spec'] = (w[::8].tolist(), (F/F.max())[::8].tolist())
        res[nm] = ws
        # halves for stability
        h = []
        for a, b in [(tacc + 10, tacc + 80), (tacc + 80, t[-1] + 1)]:
            mm = (t >= a) & (t < b); s = np.array([q['s'][0][k] for q in r]); w, F = spec(t[mm], s[mm]); h.append(peak(w, F))
        res[nm + '_halves'] = h
    out[f] = res
    print(f, {k: v for k, v in res.items() if 'spec' not in k})
json.dump(out, open('ana293.json', 'w'))
