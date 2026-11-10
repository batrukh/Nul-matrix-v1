"""(322) ідеї 1 (релікти vs швидкість народження) і 2 (шрами від злиття) + життя вихорів Аліси; зерна 3–6 з директором, знімки щоодиниці."""
import json, glob, numpy as np, scipy.ndimage as ndi
from scipy.stats import spearmanr
L = 96; ST = np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]])
NB = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
def plabel(W):
    lab, n = ndi.label(W, ST); par = list(range(n + 1))
    def f(a):
        while par[a] != a: par[a] = par[par[a]]; a = par[a]
        return a
    for (di, dj) in NB:
        B = np.roll(lab, (-di, -dj), (0, 1)); m = (lab > 0) & (B > 0) & (lab != B)
        for a, b in set(zip(lab[m].tolist(), B[m].tolist())): par[f(a)] = f(b)
    r = np.array([f(a) for a in range(n + 1)]); lab = r[lab]
    u = np.unique(lab[lab > 0]); m = np.zeros(lab.max() + 1, int); m[u] = np.arange(1, len(u) + 1); return m[lab], len(u)
def tri_ij(p): return [(int(q[1]), int(q[2])) for q in p]
def pdist(a, b):
    di = (a[0] - b[0] + L//2) % L - L//2; dj = (a[1] - b[1] + L//2) % L - L//2
    return np.hypot(di + dj/2, dj*np.sqrt(3)/2)
res = {}; R1 = []; MERGE = dict(n_new_merge=0, area_t_merge=0.0, n_new_other=0, area_t_other=0.0, near_seam=0, n_seam_test=0)
for d in sorted(glob.glob('s*/')):
    dt = json.load(open(glob.glob(d + '*_dtrack.json')[0])); fr = [q for q in dt if q.get('psi') is not None]
    T = [q['k']*0.05 for q in fr]; Ws = [np.unpackbits(np.array(q['psi'], np.uint8))[:L*L].reshape(L, L).astype(bool) for q in fr]
    labs = [plabel(W) for W in Ws]
    births = []; prev = None; owner = {}               # owner[(t, label)] = set of birth ids
    for k, (lab, n) in enumerate(labs):
        for c in range(1, n + 1):
            m = lab == c
            if prev is None or not (prev[0][m] > 0).any():
                bid = len(births); births.append(dict(t=T[k], k=k, a0=int(m.sum()))); owner[(k, c)] = {bid}
            else:
                anc = set(np.unique(prev[0][m])) - {0}; s = set()
                for a in anc: s |= owner[(k - 1, a)]
                owner[(k, c)] = s
                if len(anc) >= 2 and k + 1 < len(fr):           # злиття: нові дисклінації в наступні 5 кадрів
                    seam = m & (prev[0] == 0)
                    old = tri_ij(fr[k - 1].get('dpos', []))
                    for kk in range(k, min(k + 5, len(fr))):
                        new = [p for p in tri_ij(fr[kk].get('dpos', [])) if labs[kk][0][p] > 0 and all(pdist(p, o) > 3 for o in old)]
                        new = [p for p in new if (labs[kk][0] == labs[kk][0][np.where(m)][0]).any() and m[p]]
                        MERGE['n_new_merge'] += len(new); MERGE['area_t_merge'] += m.sum()
                        if len(new):
                            sp = np.argwhere(seam)
                            for p in new:
                                MERGE['n_seam_test'] += 1
                                if len(sp) and min(pdist(p, s_) for s_ in sp[::max(1, len(sp)//200)]) <= 4: MERGE['near_seam'] += 1
                        old = tri_ij(fr[kk].get('dpos', []))
        prev = (lab, n)
    # контроль: нові дисклінації у світах без недавнього злиття
    for k in range(1, len(fr)):
        old = tri_ij(fr[k - 1].get('dpos', [])); lab = labs[k][0]
        for p in tri_ij(fr[k].get('dpos', [])):
            if lab[p] > 0 and all(pdist(p, o) > 3 for o in old): MERGE['n_new_other'] += 1
        MERGE['area_t_other'] += (lab > 0).sum()
    # ідея 1: швидкість росту світу в перші 5 од. після народження vs густина дисклінацій у ньому при t 60, лише в світах-одинаках
    for b in births:
        k0 = b['k']; k5 = min(k0 + 5, len(fr) - 1)
        cs = [c for c in range(1, labs[k5][1] + 1) if owner.get((k5, c)) == {births.index(b)}]
        b['v'] = float(sum((labs[k5][0] == c).sum() for c in cs) - b['a0'])/5 if cs else None
    for tt in (60, 150):
        k = int(np.argmin(np.abs(np.array(T) - tt))); lab, n = labs[k]; dp = tri_ij(fr[k].get('dpos', []))
        for c in range(1, n + 1):
            o = owner[(k, c)]
            if len(o) == 1:
                b = births[list(o)[0]]; a = int((lab == c).sum())
                if b['v'] is not None and a >= 50:
                    R1.append(dict(seed=d, tt=tt, v=b['v'], tb=b['t'], area=a, nd=sum(1 for p in dp if lab[p] == c)))
    # життя вихорів Аліси (кластер у радіусі 4, пропуски ≤3 од.)
    tr = []
    for q in dt:
        if q.get('zpos') is None: continue
        tr.append((q['k']*0.05, tri_ij(q['zpos'])))
    best = 0; lives = []
    for i0, (t0, zs) in enumerate(tr):
        for z in zs:
            if i0 > 0 and any(pdist(z, x) < 4 for x in tr[i0 - 1][1]): continue
            last = t0
            for t, z2 in tr[i0:]:
                if any(pdist(z, x) < 5 for x in z2): last = t
                elif t - last > 3: break
            lives.append(last - t0)
    nd_end = len(fr[-1].get('dpos', [])); nw = labs[-1][1]
    nb = [len(b) for b in [births]]
    nmerge = 0
    res[d] = dict(births=len(births), worlds_end=nw, nd_end=nd_end, nz_end=len(fr[-1].get('zpos', [])), life_max=max(lives) if lives else 0,
                  lives_gt100=sum(1 for x in lives if x > 100), nd_t=[(round(q['k']*0.05), len(q.get('dpos', []))) for q in dt[::25]])
    print(d, {k: v for k, v in res[d].items() if k != 'nd_t'}, flush=True)
for tt in (60, 150):
    r = [x for x in R1 if x['tt'] == tt]
    if len(r) > 3:
        rho = spearmanr([x['v'] for x in r], [x['nd']/x['area'] for x in r]); rho2 = spearmanr([x['v'] for x in r], [x['nd'] for x in r])
        print('ідея 1 t', tt, 'світів', len(r), 'Спірмен v–густина', round(rho[0], 2), 'p', round(rho[1], 3), '| v–число', round(rho2[0], 2), 'p', round(rho2[1], 3))
rm = MERGE['n_new_merge']/max(MERGE['area_t_merge'], 1)*1e4; ro = MERGE['n_new_other']/max(MERGE['area_t_other'], 1)*1e4
print('ідея 2: нові дисклінації на 10⁴ вузло-кадрів — після злиття', round(rm, 2), 'загалом', round(ro, 2), MERGE)
json.dump(dict(res=res, R1=R1, MERGE=MERGE, rate_merge=rm, rate_all=ro), open('ana322.json', 'w'))
