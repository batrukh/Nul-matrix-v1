import os, sys, time, numpy as np
os.environ["NB"] = "0"; os.environ.setdefault("DELTA", "1e-6")
sys.path[:0] = ["/mnt/project-files/model/s52/r396", "/mnt/project-files/model/s52/r351", "/mnt/project-files/model/s52", "/mnt/project-files/model/s52/r463"]
import fastpatch; fastpatch.install()
import fabgen as G, rq351f as Q, kund
D = int(sys.argv[1]); L = int(sys.argv[2]); lc = float(sys.argv[3])
kund.install(D); t0 = time.time()
s, f, mod, z, psi = Q.run(L=L, seed=1, Tt=1, ell=4.0, s0=3.0, h=0.0, lc=lc, ac=0.25, check=1)
print("build", round(time.time() - t0, 1), "s; n", f.n, "m", f.m, "deg", np.unique(f.deg), "tri/node", f.ntri/f.n, "quads", len(mod.QV), "/node", len(mod.QV)/f.n,
      "loops/link", np.unique(mod.ke), "s.lc", s.lc, "eps0", mod.eps0, "kappa", mod.kappa)
# перевірка: кожна пара сусідніх вершин петлі — нитка (links() уже assert), кожен тетраедр має 6 різних ниток
s.S0 = float((np.abs(z)**2).sum()); s.m0 = s.S0/f.m
t0 = time.time(); [s.F(z, psi) for _ in range(5)]; print("F", round((time.time() - t0)/5, 3), "s")
