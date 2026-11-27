"""(470) пробна маса: зв'язувач запасу в кулі r≤R0 навколо вузла (додатковий член +μ z на нитках кулі, з тим самим законом збереження K),
стаціонарний профіль δρ(r) навколо. Модель rq351f+fastpatch (база 8.11), α = ac. python3 pm470.py s0 seed [ac L mu T]"""
import sys, os, json, time, numpy as np
R = "/mnt/project-files/model/s52"; sys.path[:0] = [R + "/r447", R + "/r396", R + "/r351", R]
import fastpatch; fastpatch.install()
import rq351f as Q
s0 = float(sys.argv[1]); seed = int(sys.argv[2]); ac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
L = int(sys.argv[4]) if len(sys.argv) > 4 else 16; MU = float(sys.argv[5]) if len(sys.argv) > 5 else 0.2; T = float(sys.argv[6]) if len(sys.argv) > 6 else 300.0
H = 0.36; DT = 0.005; R0 = 1.0
D = os.path.dirname(os.path.abspath(__file__)) + "/runs"; os.makedirs(D, exist_ok=True)
tag = f"s{s0:g}_a{ac:g}_L{L}_mu{MU:g}_s{seed}"; out = f"{D}/{tag}.json"; ckf = f"{D}/{tag}_ck.npz"
if os.path.exists(out): sys.exit()
s, f, mod, z, psi = Q.run(L=L, seed=seed, s0=s0, h=H, ac=ac, check=1)
s.S0 = float((np.abs(z)**2).sum()); s.m0 = s.S0/f.m
rng = np.random.default_rng(500 + seed); c0 = int(rng.integers(f.n))
dI = (f.ijk - f.ijk[c0] + L//2) % L - L//2; X = dI.astype(float)   # декартові відстані у вкладенні кубічної ґратки (Кун)
rn = np.linalg.norm(X, axis=1); mid = (X[f.tail] + ((X[f.head] - X[f.tail] + L//2) % L - L//2)/2); re = np.linalg.norm(mid, axis=1)
ball = re <= R0 + 1e-9
F0 = s.F
def F(z, psi, parts=False):
    if parts: return F0(z, psi, parts)
    dz, fp = F0(z, psi)
    rho = np.maximum(np.abs(z)**2, 0.05); ex = np.where(ball, MU, 0.0)*z; src = 2*(z.conjugate()*ex).real
    ks = fastpatch._krfft(s, src)
    return dz + ex - ks/(2*rho)*z, fp
s.F = F
t = 0.0; hist = []
if os.path.exists(ckf):
    c = np.load(ckf); z = c["z"]; psi = c["psi"]; t = float(c["t"]); s.S0 = float(c["S0"]); s.m0 = s.S0/f.m; hist = json.loads(str(c["hist"]))
t0 = time.time(); ST = int(round(1/DT))
def prof(z, psi):
    rho = np.abs(z)**2; rb = np.round(re*2)/2; P = {}
    for r in np.unique(rb):
        if r > L/2: continue
        sel = rb == r; P[f"{r:g}"] = [float(rho[sel].mean()), int(sel.sum())]
    pv = {}
    rr = np.round(rn)
    for r in np.unique(rr):
        if r > L/2: continue
        pv[f"{r:g}"] = float(psi[rr == r].mean())
    return P, pv
while t < T - 1e-9:
    for _ in range(ST):
        z, psi = s.step(z, psi, DT)
    t += 1.0
    if not np.isfinite(z).all(): hist.append(dict(t=t, blowup=True)); break
    if int(round(t)) % 10 == 0:
        rho = np.abs(z)**2; hist.append(dict(t=t, in_=float(rho[ball].mean()), far=float(rho[re > L/2 - 1].mean()), psimax=float(psi.max()), sec=round(time.time() - t0)))
        print(tag, hist[-1], flush=True)
        np.savez(ckf + ".tmp.npz", z=z, psi=psi, t=t, S0=s.S0, hist=json.dumps(hist)); os.replace(ckf + ".tmp.npz", ckf)
P, pv = prof(z, psi)
json.dump(dict(s0=s0, ac=ac, L=L, mu=MU, seed=seed, T=t, c0=c0, prof_rho=P, prof_psi=pv, hist=hist), open(out + ".tmp", "w")); os.replace(out + ".tmp", out)
if os.path.exists(ckf): os.remove(ckf)
