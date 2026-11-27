"""(381) правило важеля в 3D: частка ниток світу = (s0_eff − A_M)/(A_W − A_M) на кінцевих станах (374) і (375C)."""
import sys, glob, json; sys.argv = ['x']; sys.path[:0] = ['/mnt/project-files/model/s52/r351', '/mnt/project-files/model/s52']
import numpy as np, rq351f as Q
s, f, mod, z, psi = Q.run(L=16, check=1); out = []
for d in sorted(glob.glob('/mnt/project-files/model/s52/r374/a0.*_seed*')) + sorted(glob.glob('/mnt/project-files/model/s52/r375/a0.1_seed*_T600')):
    fn = glob.glob(d + '/t3*_ell4.npz')
    if not fn: continue
    c = np.load(fn[0], allow_pickle=True); A = np.abs(c['z'])**2; p = c['psi']
    W = (p[f.tail] > 0) & (p[f.head] > 0); M = ~W
    if W.sum() < 50: continue
    AW, AM, s0 = A[W].mean(), A[M].mean(), A.mean()
    r = dict(run=d.split('/')[-2] + '/' + d.split('/')[-1], s0=float(s0), AW=float(AW), AM=float(AM), fW=float(W.mean()), lever=float((s0 - AM)/(AW - AM)), dead=float((A < 0.01).mean()))
    out.append(r); print(r)
json.dump(out, open('lever381.json', 'w'), indent=1)
