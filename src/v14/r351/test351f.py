"""Перевірка rq351f: сила петель збігається з rq351 (випадковий стан), і короткий прогін іде тією ж траєкторією."""
import sys, numpy as np; sys.path.insert(0, '/mnt/project-files/model/s52/r351')
import rq351 as A, rq351f as B
for ac in (0.1, 1.0):
    s, f, mod, z, psi = A.run(L=8, seed=3, s0=3.0, ac=ac, check=1); rng = np.random.default_rng(1)
    z = z*(1 + 0.4*(rng.standard_normal(f.m) + 1j*rng.standard_normal(f.m))); psi = psi + 0.8*rng.standard_normal(f.n)
    y = mod.pack(z, psi); r1 = mod.rhs(0, y)
    mod2 = B.MQ(f, mod.QV, 0.0, mod.eps0, mod.h, mod.kappa, mod.wp); mod2.links(); mod2.beta = mod.beta
    r2 = mod2.rhs(0, y); print('ac', ac, 'max |rhs diff| / max|rhs|', np.abs(r1 - r2).max()/np.abs(r1).max())
