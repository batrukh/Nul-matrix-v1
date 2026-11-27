import numpy as np, time, scipy.sparse as sp, sys
import local60 as LC, stall43 as T, rnd51 as R, seam51 as SM, fftk
L=int(sys.argv[1]); n=int(sys.argv[2])
s = T.Sys(L=L, h=0.3, lc=1.3); f = s.fab; R.flip(f, SM.seams(f))
E3 = np.stack([f.E1, f.E2, f.E3], 1); rows=[]; cols=[]
for a, b in ((0, 1), (1, 2), (0, 2)): rows += [E3[:, a], E3[:, b]]; cols += [E3[:, b], E3[:, a]]
A = sp.csr_matrix((np.ones(sum(len(r) for r in rows)), (np.concatenate(rows), np.concatenate(cols))), shape=(f.m, f.m)); A.data[:]=1.0
P = ((sp.identity(f.m) + A)/5.0).tocsr()
ap = fftk.build(f, L, P, n)
v = np.random.default_rng(0).standard_normal(f.m)
t=time.time(); w=v.copy()
for _ in range(n): w = P @ w
t1=time.time()-t; t=time.time(); u=ap(v); t2=time.time()-t
print('m',f.m,'maxerr',np.abs(u-w).max(),'scale',np.abs(w).max(),'matvec',round(t1,4),'fft',round(t2,4))
