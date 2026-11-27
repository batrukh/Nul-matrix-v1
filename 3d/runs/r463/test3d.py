"""Звірка kund (d=3) зі старим кодом fabgen 'tet' + rq351f.quads: ті самі нитки, трикутники, петлі, P, K і сила F."""
import os, sys, numpy as np
os.environ["NB"] = "0"; os.environ.setdefault("DELTA", "1e-6")
sys.path[:0] = ["/mnt/project-files/model/s52/r396", "/mnt/project-files/model/s52/r351", "/mnt/project-files/model/s52", "/mnt/project-files/model/s52/r463"]
import fastpatch; fastpatch.install()
import fabgen as G, rq351f as Q, kund
L = 8; kw = dict(L=L, seed=3, Tt=1, ell=4.0, s0=3.0, h=0.3, lc=3.0333, ac=0.25, check=1)
old = Q.run(**kw)
sv = (G.FabricG, G.tau_cubic, G.fft_K, fastpatch._krfft, Q.quads)
kund.install(3, G.LATT["tet"]); new = Q.run(**kw)
so, fo, mo, zo, po = old; sn, fn, mn, zn, pn = new
print("m", fo.m, fn.m, "tail/head eq", (fo.tail == fn.tail).all(), (fo.head == fn.head).all())
print("tri", fo.ntri, fn.ntri, "sets eq", set(map(tuple, np.sort(fo.TV, 1))) == set(map(tuple, np.sort(fn.TV, 1))))
def cyc(QV):
    out = set()
    for r in map(np.array, QV.tolist()):
        out.add(min(tuple(np.roll(w, k)) for k in range(4) for w in (r, r[::-1])))
    return out
print("quads", len(mo.QV), len(mn.QV), "sets eq", cyc(mo.QV) == cyc(mn.QV))
print("P diff", abs(so.P - sn.P).max(), "deg", fo.deg[0], fn.deg[0], "lc", so.lc, sn.lc, "beta eq", np.allclose(np.sort(mo.beta), np.sort(mn.beta)))
for s in (so, sn): s.S0 = float((np.abs(zo)**2).sum()); s.m0 = s.S0/fo.m
rng = np.random.default_rng(1); z = zo*(1 + 0.3*rng.standard_normal(fo.m)) + 0.2j*rng.standard_normal(fo.m); psi = po + 0.5*rng.standard_normal(fo.n)
Fo = so.F(z, psi); Fn = sn.F(z, psi)
print("F z rel", np.abs(Fo[0] - Fn[0]).max()/np.abs(Fo[0]).max(), "F psi rel", np.abs(Fo[1] - Fn[1]).max()/np.abs(Fo[1]).max())
src = rng.standard_normal(fo.m); print("K rel", np.abs(so.K(src) - sn.K(src)).max()/np.abs(so.K(src)).max())
