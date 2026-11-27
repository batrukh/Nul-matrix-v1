import os, sys, numpy as np
os.environ["NB"] = "0"; R = "/mnt/project-files/model/s52"; sys.path[:0] = [R + "/r396", R + "/r351", R]
import fastpatch; fastpatch.install(); import rq351f as Q
ac, h, seed = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
s, f, mod, z, psi = Q.run(L=16, seed=seed, s0=3.0, ell=4.0, ac=ac, h=h, check=1)
s.S0 = float((np.abs(z)**2).sum()); s.m0 = s.S0/f.m; t = 0.0
while t < 6 - 1e-9: z, psi = s.step(z, psi, 0.005); t += 0.005
print(ac, h, seed, round(float(psi.mean()), 4), flush=True)
