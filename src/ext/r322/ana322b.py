"""(322) контроль до ідеї 2: нові дисклінації на НОВІЙ площі світу — шов злиття vs звичайний фронт росту; виживання через 10 і 50 од."""
import json, glob, numpy as np
exec(open('ana322.py').read().split('res = {}')[0])
out = dict(seam=[0, 0, 0, 0], front=[0, 0, 0, 0])      # [нових дисклінацій, нових вузлів, живих через 10, через 50]
R1 = []
for d in sorted(glob.glob('s*/')):
    dt = json.load(open(glob.glob(d + '*_dtrack.json')[0])); fr = [q for q in dt if q.get('psi') is not None]
    T = [q['k']*0.05 for q in fr]; labs = [plabel(np.unpackbits(np.array(q['psi'], np.uint8))[:L*L].reshape(L, L).astype(bool)) for q in fr]
    D = [tri_ij(q.get('dpos', [])) for q in fr]
    def alive(p, k):
        return k < len(fr) and any(pdist(p, x) <= 3 for x in D[k])
    births = 0
    for k in range(1, len(fr)):
        lab, n = labs[k]; pl = labs[k - 1][0]
        for c in range(1, n + 1):
            m = lab == c; anc = set(np.unique(pl[m])) - {0}
            if not anc: births += 1; continue
            newA = m & (pl == 0); kind = 'seam' if len(anc) >= 2 else 'front'
            out[kind][1] += int(newA.sum())
            for p in D[k]:
                if newA[p] and all(pdist(p, o) > 3 for o in D[k - 1]):
                    out[kind][0] += 1; out[kind][2] += alive(p, k + 10); out[kind][3] += alive(p, k + 50)
    # ідея 1 простіше: число народжень світів (швидкість «закипання») vs дисклінацій на площу при t 150 і 300
    k150 = int(np.argmin(np.abs(np.array(T) - 150)))
    A = [(labs[k][0] > 0).sum() for k in (k150, len(fr) - 1)]
    t_half = T[next(k for k in range(len(fr)) if (labs[k][0] > 0).sum() > 0.5*A[1])]
    R1.append(dict(seed=d, births=births, t_half=t_half, nd150=len(D[k150]), nd300=len(D[-1]), A300=int(A[1])))
for k, v in out.items(): print(k, 'нових', v[0], 'на 1000 нових вузлів', round(1000*v[0]/max(v[1], 1), 2), 'живих через 10', v[2], 'через 50', v[3])
for r in R1: print(r)
json.dump(dict(out=out, R1=R1), open('ana322b.json', 'w'))
