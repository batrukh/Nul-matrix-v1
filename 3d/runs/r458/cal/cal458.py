"""Калібрування затвора: open — eps 0,25 скрізь (швидкість фронту v_P); shut — 0,25 лише в кристалі (c>0,5), 0,20 деінде (фронт має стояти).
python3 cal458.py MODE EOFF seed TT"""
import sys, json, numpy as np; sys.path.insert(0, "/mnt/project-files/model/s52/r458"); import pfcB as P
mode, eoff, seed, TT = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
rng = np.random.default_rng(seed); cen = np.array([P.LB/2]*3); rad = 3.5*P.LB/16
psi = P.seedball(rng, cen, rad); x = np.arange(P.NG)*P.DX; X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
r = np.sqrt((X - cen[0])**2 + (Y - cen[1])**2 + (Z - cen[2])**2); hist = []; t = 0; dt = 0.5
while t < TT:
    c = P.cryst(psi); cr = c > 0.5
    epsx = np.full(psi.shape, P.EPS0) if (mode == "open" or t < 60) else np.where(cr, P.EPS0, eoff)   # shut: перші t60 — дозрівання (open)
    psi = P.step(psi, epsx, dt, 20); t += 20*dt
    V = cr.sum()*P.DX**3; Rk = (3*V/(4*np.pi))**(1/3)*16/P.LB   # радіус у кроках Куна
    hist.append(dict(t=t, Rkun=round(float(Rk), 3), frac=round(float(cr.mean()), 4), cmax=round(float(c.max()), 3))); print(hist[-1], flush=True)
    if cr.mean() > 0.97: break
json.dump(hist, open(f"/mnt/project-files/model/s52/r458/cal/{mode}_{eoff}_s{seed}.json", "w"))
