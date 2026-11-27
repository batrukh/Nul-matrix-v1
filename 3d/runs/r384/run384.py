import sys, os, shutil; sys.path[:0] = ['/mnt/project-files/model/s52/r351', '/mnt/project-files/model/s52']
import rq351f as Q
_rep = os.replace; k = [0]
def rep(a, b):
    _rep(a, b)
    if b.endswith('_ck.npz'):
        import numpy as np; t = float(np.load(b, allow_pickle=True)['t']); shutil.copy(b, f'snap_{int(round(t)):04d}.npz')
Q.os.replace = rep
Q.run(L=16, s0=2.95, seed=7, Tt=230.0, ell=4.0, ck=5.0, ac=0.2)
