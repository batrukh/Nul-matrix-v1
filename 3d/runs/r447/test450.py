import sys, time, numpy as np
sys.argv = ['x']; sys.path.insert(0, '/mnt/project-files/model/s52/r447')
import rq447 as W, kcheb
for lat in ('/mnt/project-files/model/s52/r448/runs/R_e0.25_s1.npz', '/mnt/project-files/model/s52/r448/runs/R_e0.25_s3.npz'):
    s, f, mod, z, psi = W.run(lat=lat, seed=1, check=1, ac=0.1)
    K2 = kcheb.make_K(s.P, s.n); print(lat[-12:], K2.info)
    rng = np.random.default_rng(1)
    for _ in range(3):
        x = rng.standard_normal(f.m); a = s.K(x); b = K2(x); print(' rel', np.abs(a - b).max()/np.abs(x).max())
    x = rng.standard_normal(f.m)
    for fn, nm in ((s.K, 'old'), (K2, 'new')):
        fn(x); t = time.time(); [fn(x) for _ in range(20)]; print(' ', nm, 'ms', round((time.time() - t)/20*1e3, 1))
