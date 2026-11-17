"""Перевірка: рівномірний стан z = w (усі петлі дійсні додатні) — чи стаціонарний у 3D-тетраедрах; рівноважна |z|² для β 1,3."""
import sys; sys.path.insert(0, '/mnt/project-files/model/s52/r351'); sys.path.insert(0, '/mnt/project-files/model/s52')
import numpy as np, rq351 as Q
s, f, mod, z, psi = Q.run(L=8, check=1, beta=1.3)
psi[:] = psi.min()*0 + np.roots([1,0,-1,-0.3])[np.isreal(np.roots([1,0,-1,-0.3]))].real.min()
for w2 in [3.5, 4.0, 4.54, 5.0]:
    z = np.sqrt(w2)*np.ones(f.m)+0j
    d = mod.rhs(0, mod.pack(z, psi)); dz, dp = mod.unpack(d)
    print(w2, 'dz/z mean', float((dz/z).real.mean()), 'spread', float(np.std((dz/z).real)))
