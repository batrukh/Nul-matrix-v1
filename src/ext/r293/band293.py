"""(293) смуга дихання: пік спектра в 0,25–0,55 і загасання амплітуди (смуговий фільтр) для різниці «з поштовхом − без»."""
import json, numpy as np
def band(t, s, lo=0.25, hi=0.55):
    dt = t[1] - t[0]; F = np.fft.rfft(s - s.mean()); w = 2*np.pi*np.fft.rfftfreq(len(s), dt); F[(w < lo) | (w > hi)] = 0; return np.fft.irfft(F, len(s))
def peak(t, s, lo=0.25, hi=0.55):
    x = (s - np.polyval(np.polyfit(t, s, 2), t))*np.hanning(len(s)); dt = t[1] - t[0]
    F = np.abs(np.fft.rfft(x, 64*len(x))); w = 2*np.pi*np.fft.rfftfreq(64*len(x), dt); m = (w > lo) & (w < hi)
    i = np.argmax(F[m]); return float(w[m][i]), float(F[m][i]/F[(w > 0.1)].max())
out = {}
r = json.load(open('mv_g0_acc0.json'))['rec']; t = np.array([q['t'] for q in r]); s = np.array([q['s'][0][0] for q in r])
for a, b in [(5, 45), (5, 75), (5, 150)]:
    m = (t > a) & (t < b); print('rest', a, b, peak(t[m], s[m]))
bs = band(t, s); out['rest'] = dict(t=t.tolist(), band=bs.tolist(), v=0.0)
for g in ['0.02', '0.04']:
    A = json.load(open(f'kk_g{g}_k0.1.json'))['rec']; B = json.load(open(f'kk_g{g}_k0.json'))['rec']; n = min(len(A), len(B)); t = np.array([q['t'] for q in A[:n]])
    s = np.array([q['s'][0][0] for q in A[:n]]) - np.array([q['s'][0][0] for q in B[:n]])
    for a, b in [(41, 81), (41, 111), (41, 181)]:
        m = (t > a) & (t < b); print(g, a, b, peak(t[m], s[m]))
    m = t > 41; bs = band(t[m], s[m]); env = [float(np.std(bs[(t[m] >= x) & (t[m] < x + 20)])) for x in range(41, 161, 20)]
    print('  env', [round(e, 6) for e in env])
    out[g] = dict(t=t[m].tolist(), band=bs.tolist())
json.dump(out, open('band293.json', 'w'))
