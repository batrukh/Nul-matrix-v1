import json, glob, numpy as np, scipy.ndimage as ndi
res = {}
for d in ['s1_d0', 's1_d1', 's2_d0', 's2_d1']:
    g = json.load(open(glob.glob(d + '/*_g104.json')[0]))['track']; dt = json.load(open(glob.glob(d + '/*_dtrack.json')[0]))
    z = np.load(glob.glob(d + '/*_g104.npz')[0]); psi = z['psi'].reshape(96, 96); W = psi > 0.3
    lab, nw = ndi.label(W); sz = sorted(np.bincount(lab.ravel())[1:].tolist(), reverse=True)
    t = [q['t'] for q in g]; wt = [q['wtri'] for q in g]; ncl = [q['ncl'] for q in g]
    first = next((q['t'] for q in g if q['wtri'] > 0), None)
    P = z['P']; t2 = np.trace(P @ P, axis1=-2, axis2=-1)
    r = dict(nw=nw, sizes=sz[:8], area=int(W.sum()), t_first=first, ncl_mean=float(np.mean(ncl)), ncl_max=int(max(ncl)), ncl_end=ncl[-1],
             t2w=float(t2[W].mean()) if W.any() else None, wt=wt, t=t, ncl=ncl, dtr=dt)
    if 'n' in z.files:
        n = z['n']; w, V = np.linalg.eigh(P); ne = np.einsum('xyij,xyi->xyj', V, n)**2
        r['n_ev'] = np.round(ne[W].mean(0), 3).tolist(); r['nd'] = [q['nd'] for q in dt]; r['nd_end'] = dt[-1]['nd']
    res[d] = r
    print(d, {k: v for k, v in r.items() if k not in ('wt', 't', 'ncl', 'dtr', 'nd')})
json.dump(res, open('ana321.json', 'w'))
