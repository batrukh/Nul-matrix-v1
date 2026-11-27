"""(422) Чи має кипіння Матриці власну довжину? Ріст малих збурень однорідної Матриці по |k| (Кун 3D, база 8.11, fastpatch).
python3 lin422.py L s0 seed ac  -> spec422_L{L}_s{s0}_seed{seed}.npz (час, потужність δψ і δ|z|² за оболонками |k|)"""
import os, sys, numpy as np
os.environ["NB"] = "0"
R = "/mnt/project-files/model/s52"; sys.path[:0] = [R + "/r396", R + "/r351", R]
import fastpatch; fastpatch.install()
import rq351f as Q
L = int(sys.argv[1]); s0 = float(sys.argv[2]); seed = int(sys.argv[3]); ac = float(sys.argv[4]); Tmax = float(sys.argv[5]) if len(sys.argv) > 5 else 15
out = f"spec422_L{L}_s{s0:g}_seed{seed}_ac{ac:g}.npz"
if os.path.exists(out): sys.exit(0)
s, f, mod, z, psi = Q.run(L=L, seed=seed, s0=s0, ell=4.0, ac=ac, check=1)
rng = np.random.default_rng(1000 + seed); EPS = 1e-6
z0 = np.sqrt(s0)*np.ones(f.m) + 0j; p0 = psi.mean()*0 + np.roots([1, 0, -1, -0.3])[np.isreal(np.roots([1, 0, -1, -0.3]))].real.min()
z = z0*(1 + EPS*(rng.standard_normal(f.m) + 1j*rng.standard_normal(f.m))); psi = p0 + EPS*rng.standard_normal(f.n)
s.S0 = float((np.abs(z)**2).sum()); s.m0 = s.S0/f.m
I = f.ijk; kk = 2*np.pi*np.fft.fftfreq(L); KX, KY, KZ = np.meshgrid(kk, kk, kk, indexing="ij")
kabs = np.sqrt(4*(np.sin(KX/2)**2 + np.sin(KY/2)**2 + np.sin(KZ/2)**2))        # ґраткове |k|
edges = np.linspace(0, kabs.max() + 1e-9, 13); sh = np.digitize(kabs, edges) - 1
def spec(v):
    g = np.zeros((L, L, L)); g[I[:, 0], I[:, 1], I[:, 2]] = v - v.mean(); P = np.abs(np.fft.fftn(g))**2
    return np.bincount(sh.ravel(), P.ravel(), len(edges))[:len(edges) - 1]/np.maximum(np.bincount(sh.ravel(), None, len(edges))[:len(edges) - 1], 1)
dt = 0.005; t = 0.0; T, SP, SZ, AMP = [], [], [], []
while t < Tmax - 1e-9:
    for _ in range(int(round(0.25/dt))): z, psi = s.step(z, psi, dt); t += dt
    A2n = (np.bincount(f.tail, np.abs(z)**2, f.n) + np.bincount(f.head, np.abs(z)**2, f.n))/f.deg
    T.append(t); SP.append(spec(psi)); SZ.append(spec(A2n)); AMP.append(float(np.std(psi)))
    if AMP[-1] > 0.05: break
np.savez(out, t=np.array(T), sp=np.array(SP), sz=np.array(SZ), amp=np.array(AMP), kc=0.5*(edges[1:] + edges[:-1]))
print(out, "t_end", t, "amp", AMP[-1], flush=True)
