"""(322) звідки дисклінації, що дожили до t 300: слід назад кадр за кадром (найближча ≤3), клас місця появи."""
import json, glob, numpy as np
exec(open('ana322.py').read().split('res = {}')[0])
cnt = dict(seam=0, front=0, birth=0, inside=0); per = {}
for d in sorted(glob.glob('s*/')):
    dt = json.load(open(glob.glob(d + '*_dtrack.json')[0])); fr = [q for q in dt if q.get('psi') is not None]
    labs = [plabel(np.unpackbits(np.array(q['psi'], np.uint8))[:L*L].reshape(L, L).astype(bool)) for q in fr]
    D = [tri_ij(q.get('dpos', [])) for q in fr]; per[d] = []
    for p in D[-1]:
        k = len(fr) - 1; q = p
        while k > 0:
            c = [x for x in D[k - 1] if pdist(q, x) <= 3]
            if not c: break
            q = min(c, key=lambda x: pdist(q, x)); k -= 1
        lab, n = labs[k]; pl = labs[k - 1][0] if k > 0 else np.zeros((L, L), int); m = lab == lab[q]
        anc = set(np.unique(pl[m])) - {0}
        kind = 'birth' if not anc else ('inside' if pl[q] > 0 else ('seam' if len(anc) >= 2 else 'front'))
        cnt[kind] += 1; per[d].append((kind, round(fr[k]['k']*0.05)))
    print(d, per[d])
print(cnt); json.dump(dict(cnt=cnt, per=per), open('origin322.json', 'w'))
