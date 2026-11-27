import numpy as np, time, sys
import stall43 as T, rnd51 as R, seam51 as SM, nbforce as NB
L=int(sys.argv[1])
s = T.Sys(L=L, h=0.3, lc=1.3); f = s.fab; R.flip(f, SM.seams(f))
rng=np.random.default_rng(3)
z = np.sqrt(3.5)*(1+0.3*(rng.standard_normal(f.m)+1j*rng.standard_normal(f.m))); psi = rng.uniform(-1.3,1.2,f.n); s.m0=3.4
a = T.Sys.F(s, z, psi); b = NB.force(s, z, psi)
print('dz relerr', np.abs(a[0]-b[0]).max()/np.abs(a[0]).max(), 'dpsi relerr', np.abs(a[1]-b[1]).max()/np.abs(a[1]).max(), 'bit-equal', (a[0]==b[0]).mean(), (a[1]==b[1]).mean())
t=time.time()
for _ in range(20): T.Sys.F(s, z, psi)
t1=(time.time()-t)/20; t=time.time()
for _ in range(20): NB.force(s, z, psi)
t2=(time.time()-t)/20
print('numpy',round(t1,4),'numba',round(t2,4),'x',round(t1/t2,1))
