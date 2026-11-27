"""an83: (83) анізотропія інтенсивності ниток за кутом нитки (декартово), правильна і випадкова сітка.
A3 — (max − min)/середнє за трьома секторами кутів (центри 0°, 60°, 120°, ±30°); H = 2|⟨I e^{2iθ}⟩|/⟨I⟩ — друга гармоніка, θ0 — її напрям."""
import numpy as np, sys, json
import simNlib as N, rnd51 as R
def geom(rnd, gseed, L=96):
    if rnd: f = R.RandFabric(L, gseed, rnd); d = R.disp(f); c = np.c_[d[:, 0], d[:, 1]*np.sqrt(3)/2]
    else:
        f = N.Fabric(L); d = f.xy[f.head] - f.xy[f.tail]; d = (d + L//2) % L - L//2; c = np.c_[d[:, 0] + d[:, 1]/2, d[:, 1]*np.sqrt(3)/2]
    return f, np.degrees(np.arctan2(c[:, 1], c[:, 0])) % 180
def an(fn, rnd=0.0, gseed=1):
    f, th = geom(rnd, gseed); s = np.load(fn); I = np.abs(s["z"])**2; psi = s["psi"]
    inw = (psi[f.tail] > 0) & (psi[f.head] > 0); out = (psi[f.tail] <= 0) & (psi[f.head] <= 0); res = {}
    sec = np.round(th/60).astype(int) % 3
    for lab, m in (("у світах", inw), ("поза", out)):
        if m.sum() < 30: res[lab] = None; continue
        mm = np.array([I[m & (sec == k)].mean() for k in range(3)]); ph = np.exp(2j*np.radians(th[m]))
        c2 = (I[m]*ph).mean(); H = 2*abs(c2)/I[m].mean(); th0 = np.degrees(np.angle(c2))/2 % 180
        res[lab] = dict(sec=np.round(mm, 2).tolist(), A3=round(float((mm.max() - mm.min())/mm.mean()), 3), H=round(float(H), 3), th0=round(float(th0), 1), n=int(m.sum()))
    return res
if __name__ == "__main__":
    for a in sys.argv[1:]:
        fn, rnd, gs = a.split(","); print(fn, an(fn, float(rnd), int(gs)))
