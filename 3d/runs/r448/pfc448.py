"""pfc448 (21.11, (448)): затравковий ріст ОЦК у метастабільній рідині. PFC (421), psibar -0,30, ящик 13 комірок ОЦК, сітка 104^3.
eps у вікні 0,22-0,27 (рідина лінійно стійка: спінодаль 3psibar^2=0,27; ОЦК вигідніша з eps~0,22).
mode Z: куля ОЦК радіуса R комірок у випадковому місці (орієнтація = осі ящика); mode N: контроль без зародка; mode R: зародок випадково повернутий (контроль 2.10: орієнтація не задана ящиком). Шум старту 0,02 (як (447)).
Запуск: python3 pfc448.py MODE SEED EPS TT -> runs/{MODE}_e{EPS}_s{SEED}.npz/json; ck кожні t500; рання зупинка: ОЦК>=95% двічі поспіль."""
import sys, os, json, time, numpy as np
R0 = "/mnt/project-files/model/s52"; sys.path[:0] = [R0 + "/r421", R0 + "/r447"]
from pfc import kgrid, step, energy, A_BCC; import classify as C; from pfc447 import peaks
import fabpts as F
CELLS, NG, RAD = 13, 104, 2.0
def seedball(psi, Lb, rng, rot=False):
    n = psi.shape[0]; x = np.arange(n)*Lb/n; X, Y, Z = np.meshgrid(x, x, x, indexing='ij'); q = 2*np.pi/A_BCC
    c = rng.uniform(0, Lb, 3); d = [((V - ci + Lb/2) % Lb) - Lb/2 for V, ci in zip((X, Y, Z), c)]
    r = np.sqrt(d[0]**2 + d[1]**2 + d[2]**2); m = 0.5*(1 - np.tanh((r - RAD*A_BCC)/2.0))
    Rm = np.eye(3)
    if rot:
        from scipy.spatial.transform import Rotation; Rm = Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix()
    V6 = np.array([(1,1,0),(1,-1,0),(0,1,1),(0,1,-1),(1,0,1),(1,0,-1)], float) @ Rm.T
    p = 0.5*sum(np.cos(q*(v[0]*d[0] + v[1]*d[1] + v[2]*d[2])) for v in V6)
    psi = psi + m*0.3*p/3.0; return psi - psi.mean() + (-0.30), c.tolist()
def measure(psi, Lb):
    P = peaks(psi, Lb, -0.30); Ls = np.array([Lb]*3)
    fr, _ = C.cna(P, Ls) if len(P) > 20 else ({'bcc': 0.0}, None)
    d14 = float(np.mean(np.asarray(F.fromPoints(P, Lb).deg) == 14)) if len(P) > 100 else 0.0
    return P, dict(npk=len(P), deg14=d14, **fr)
def main(mode, seed, eps, TT):
    out = f"runs/{mode}_e{eps}_s{seed}"; ckf = out + "_ck.npz"
    if os.path.exists(out + ".json"): return
    Lb = CELLS*A_BCC; Ls = (Lb,)*3; k2 = kgrid((NG,)*3, Ls); dt = 0.5; chunk = 200
    if os.path.exists(ckf):
        c = np.load(ckf, allow_pickle=True); psi = c["psi"].astype(float); s0 = int(c["s"]); hist = list(c["hist"]); cen = list(c["cen"])
    else:
        rng = np.random.default_rng(1000 + seed); psi = -0.30 + 0.02*rng.standard_normal((NG,)*3); psi += -0.30 - psi.mean(); s0 = 0; cen = []
        if mode in ("Z", "R"): psi, cen = seedball(psi, Lb, rng, rot=(mode == "R"))
        hist = []
    t0 = time.time(); nst = int(TT/dt); good = 0
    for s in range(s0, nst, chunk):
        psi = step(psi, k2, eps, dt, chunk); t = (s + chunk)*dt
        if int(t) % 500 == 0:
            P, m = measure(psi, Lb); hist.append(dict(t=t, F=energy(psi, k2, eps), **m)); print(mode, seed, eps, hist[-1], flush=True)
            np.savez(ckf + ".tmp.npz", psi=psi.astype(np.float32), s=s + chunk, hist=np.array(hist, dtype=object), cen=np.array(cen)); os.replace(ckf + ".tmp.npz", ckf)
            good = good + 1 if m.get('bcc', 0) >= 0.95 else 0
            if good >= 2: break
    P, m = measure(psi, Lb)
    np.savez(out + ".npz", P=P, box=Lb, psi=psi.astype(np.float32))
    json.dump(dict(mode=mode, seed=seed, eps=eps, cen=cen, final=m, hist=hist, sec=time.time() - t0), open(out + ".json", "w"), indent=1)
    if os.path.exists(ckf): os.remove(ckf)
if __name__ == "__main__": main(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
