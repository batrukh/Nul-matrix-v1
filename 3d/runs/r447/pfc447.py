"""pfc447 (20.11, (447)): ґратка з поля для світу — PFC (421) у ящику 13 комірок ОЦК (≈4394 вузли ≈ Кун L16 4096), сітка 104³ (8 точок на комірку).
prot S — повільне охолодження (ε 0,25→0,30 за t4000, далі t1000 при 0,30), як (421) S; prot Q — різке (ε 0,30 одразу з шуму 0,02, t3000), як (421) Q.
Вузли = максимуми ψ з параболічним уточненням до частки кроку сітки. Класифікація CNA — classify.py з (421). Контрольні точки кожні t500.
Запуск: python3 pfc447.py S|Q SEED → runs/lat_{prot}_s{seed}.npz (P, box, lab) + .json"""
import sys, os, json, time, numpy as np
R = "/mnt/project-files/model/s52"; sys.path[:0] = [R + "/r421"]
from pfc import kgrid, step, energy, A_BCC; import classify as C
CELLS, NG = 13, 104
def peaks(psi, Lb, pb):
    n = psi.shape[0]; m = psi > pb
    for d in [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)]: m &= psi > np.roll(psi, d, (0, 1, 2))
    I = np.argwhere(m); off = np.zeros(I.shape)
    for ax in range(3):
        e = np.zeros(3, int); e[ax] = 1
        fp = psi[tuple(((I + e) % n).T)]; fm = psi[tuple(((I - e) % n).T)]; f0 = psi[tuple(I.T)]
        off[:, ax] = 0.5*(fm - fp)/(fm - 2*f0 + fp)
    return ((I + off)*Lb/n) % Lb
def main(prot, seed):
    out = f"runs/lat_{prot}_s{seed}"; os.makedirs("runs", exist_ok=True)
    if os.path.exists(out + ".npz"): return
    Lb = CELLS*A_BCC; Ls = (Lb,)*3; k2 = kgrid((NG,)*3, Ls); pb = -0.30; dt = 0.5; ckf = out + "_ck.npz"
    sched = [(0.30, 3000.0)] if prot == "Q" else [("ramp", 4000.0), (0.30, 1000.0)]
    tot = sum(T for _, T in sched); nst = int(tot/dt); chunk = 200
    if os.path.exists(ckf): c = np.load(ckf, allow_pickle=True); psi = c["psi"].astype(float); s0 = int(c["s"]); hist = list(c["hist"]); print("resume", s0*dt, flush=True)
    else:
        rng = np.random.default_rng(seed); psi = pb + 0.02*rng.standard_normal((NG,)*3); psi += pb - psi.mean(); s0 = 0; hist = []
    t0 = time.time(); T1 = sched[0][1]/dt
    for s in range(s0, nst, chunk):
        if prot == "S" and s < T1: eps = 0.25 + 0.05*(s + chunk/2)/T1
        else: eps = 0.30
        psi = step(psi, k2, eps, dt, chunk); t = (s + chunk)*dt
        if int(t) % 500 == 0:
            P = peaks(psi, Lb, pb); fr, _ = C.cna(P, np.array(Ls)) if len(P) > 20 else ({}, None)
            hist.append(dict(t=t, eps=eps, F=energy(psi, k2, eps), npk=len(P), **fr)); print(prot, seed, hist[-1], flush=True)
            np.savez(ckf + ".tmp.npz", psi=psi.astype(np.float32), s=s + chunk, hist=np.array(hist, dtype=object)); os.replace(ckf + ".tmp.npz", ckf)
    P = peaks(psi, Lb, pb); fr, lab = C.cna(P, np.array(Ls))
    np.savez(out + ".npz", P=P, box=Lb, lab=lab)
    json.dump(dict(prot=prot, seed=seed, cells=CELLS, ng=NG, final=fr, F=energy(psi, k2, 0.30), npk=len(P), hist=hist, sec=time.time() - t0), open(out + ".json", "w"), indent=1)
    os.remove(ckf)
if __name__ == "__main__": main(sys.argv[1], int(sys.argv[2]))
