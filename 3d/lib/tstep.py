import numpy as np, time
import stall43 as T
s = T.Sys(L=96, h=0.3, lc=1.3); f=s.fab
z = np.sqrt(3.5)*(1+0.01*np.random.default_rng(1).standard_normal(f.m))+0j; psi=-1.1+0.01*np.random.default_rng(2).standard_normal(f.n)
t=time.time()
for _ in range(20): T.Sys.F(s, z, psi)
print('base F', (time.time()-t)/20)
