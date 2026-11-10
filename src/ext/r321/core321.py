import sys, glob, json, numpy as np
sys.path.insert(0, 'code'); import gf104 as GFm, gauge91 as G
out = {}
for d in ['s1_d1', 's2_d1']:
    z = np.load(glob.glob(d + '/*_g104.npz')[0]); L = 96
    gf = GFm.GF.__new__(GFm.GF); gf.L = L; gf.P, gf.U, gf.n = z['P'], z['U'], z['n']
    dd = gf.ddefects(); psi = z['psi'].reshape(L, L)
    det, wm = gf.defects(z['psi'], aware=True)
    w, V = np.linalg.eigh(gf.P); ne = np.einsum('xyij,xyi->xyj', V, gf.n)**2
    sp = w[..., 2] - w[..., 1]                       # розщеплення трійки директором
    pts = np.argwhere((dd < 0) & wm); rows = []
    for (u, i, j) in pts:
        rr = []
        for R in range(0, 9):
            m = np.zeros((L, L), bool)
            for a in range(-R, R + 1):
                for b in range(-R, R + 1):
                    if max(abs(a), abs(b)) == R: m[(i + a) % L, (j + b) % L] = True
            rr.append((round(float(sp[m].mean()), 3), round(float(ne[m][:, 2].mean()), 3)))
        z2near = bool(((det < 0) & wm)[:, max(i-3,0):i+4, max(j-3,0):j+4].any())
        rows.append(dict(ij=[int(i), int(j)], psi=round(float(psi[i, j]), 2), z2near=z2near, ring=rr))
    ok = psi > 0.3; deep = ok & (np.abs(G.sh(psi, (3, 0)) ) > 0.3)
    out[d] = dict(split_world=float(sp[ok].mean()), rows=rows)
    print(d, 'розщеплення в світі', round(float(sp[ok].mean()), 3))
    for r in rows: print(r['ij'], r['psi'], r['z2near'], r['ring'][0], r['ring'][2], r['ring'][5], r['ring'][8])
json.dump(out, open('core321.json', 'w'))
