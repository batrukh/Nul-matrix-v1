"""(423) Ефективна температура кипіння Матриці (FDT). python3 fdt423.py s0 ac seed [tw] -> fdt423_s{s0}_ac{ac}_seed{seed}.json"""
import os, sys, json, types, numpy as np
os.environ["NB"] = "0"
R = "/mnt/project-files/model/s52"; sys.path[:0] = [R + "/r396", R + "/r351", R]
import fastpatch; fastpatch.install()
import rq351f as Q, local60 as LC
s0 = float(sys.argv[1]); ac = float(sys.argv[2]); seed = int(sys.argv[3]); tw = float(sys.argv[4]) if len(sys.argv) > 4 else 6.0
out = f"runs/fdt423_s{s0:g}_ac{ac:g}_seed{seed}.json"
if os.path.exists(out): sys.exit(0)
s, f, mod, z, psi = Q.run(L=16, seed=seed, s0=s0, ell=4.0, ac=ac, check=1)
s.S0 = float((np.abs(z)**2).sum()); s.m0 = s.S0/f.m
F0 = LC.Loc.F; HF = {"v": None}
def F(self, z, psi, parts=False):
    dz, fp = F0(self, z, psi)
    return (dz, fp) if HF["v"] is None else (dz, fp + HF["v"])
s.F = types.MethodType(F, s)
dt = 0.005; t = 0.0
while t < tw - 1e-9: z, psi = s.step(z, psi, dt); t += dt
A2n = lambda z: (np.bincount(f.tail, np.abs(z)**2, f.n) + np.bincount(f.head, np.abs(z)**2, f.n))/f.deg
eps = np.where(np.random.default_rng(500 + seed).random(f.n) < 0.5, -1.0, 1.0)
TAU = np.arange(1, 13)*0.25
def branch(dh):
    HF["v"] = None if dh == 0 else dh*eps; zz, pp = z.copy(), psi.copy(); res = []
    for k in range(len(TAU)):
        for _ in range(50): zz, pp = s.step(zz, pp, dt)
        res.append((pp.copy(), A2n(zz)))
    HF["v"] = None; return res
b0 = branch(0.0); b1 = branch(1e-3); b2 = branch(2e-3)
d0 = psi - psi.mean(); a0 = A2n(z); a0 = a0 - a0.mean()
r = dict(s0=s0, ac=ac, seed=seed, tw=tw, tau=TAU.tolist(), var=float(d0.var()), psimean=float(psi.mean()), A2=float(a0.var()), Delta=[], chi1=[], chi2=[], DeltaA=[], chiA1=[])
for (p0, q0), (p1, q1), (p2, q2) in zip(b0, b1, b2):
    d = p0 - p0.mean(); r["Delta"].append(float(0.5*np.mean((d - d0)**2)))
    r["chi1"].append(float(np.mean(eps*(p1 - p0))/1e-3)); r["chi2"].append(float(np.mean(eps*(p2 - p0))/2e-3))
    qq = q0 - q0.mean(); r["DeltaA"].append(float(0.5*np.mean((qq - a0)**2))); r["chiA1"].append(float(np.mean(eps*(q1 - q0))/1e-3))
os.makedirs("runs", exist_ok=True); json.dump(r, open(out, "w"))
print(out, "Teff~", r["Delta"][-1]/r["chi1"][-1], "lin", r["chi2"][-1]/r["chi1"][-1], flush=True)
