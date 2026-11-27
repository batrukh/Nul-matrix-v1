"""chir77: хіральність при вихорах циклічного до-часу (77). Стріли відтворюються точно (поле нерухоме).
Чиста ⟨sin h⟩; біля вихорів «+» і «−» окремо (трикутники в межах r ≤ 6 і r ≤ 15 від ядра); світи біля вихорів."""
import numpy as np, os, glob
os.environ.setdefault("DELTA", "1e-6")
import simNlib as N, rnd51 as R, local60 as A
L = 96
def fab(vort, seed, g=0.02):
    f = N.Fabric(L); cores = A.vortex_cores(L, vort, seed); dth = A.vortex_dtheta(f, L, cores)
    x_, y_ = f.xy[f.tail].T; x2, y2 = f.xy[f.head].T; Gg = g*(((x2-x_+L//2) % L-L//2)+0.31*((y2-y_+L//2) % L-L//2))
    R.flip(f, dth+Gg < 0); return f, cores
for vort, seed, ac in ((2, 1, 1), (4, 1, 1), (2, 2, 1), (2, 1, -1)):
    fn = f"lc_s03.5_n100_L96_s{seed}_g0.02" + (f"_ac{ac:g}" if ac != 1 else "") + f"_v{vort}g0.02.npz"
    if not os.path.exists(fn): print("no", fn); continue
    d = np.load(fn); z, psi = d["z"], d["psi"]; f, cores = fab(vort, seed)
    l1, l2, l3 = f.tri_links(z); s = np.sin(np.angle(l1*l2*l3)); sig = np.sign(f.S1+f.S2+f.S3)
    tc = f.xy[f.TV].astype(float)                              # центр трикутника (наближено, з урахуванням тора)
    ref = tc[:, 0:1, :]; tc = ref + ((tc - ref + L/2) % L - L/2); cen = tc.mean(1)
    res = {}
    for q in (1, -1):
        for R_ in (6, 15):
            m = np.zeros(f.ntri, bool); wn = 0
            for (a, b, qq) in cores:
                if qq != q: continue
                dd = np.hypot((cen[:, 0]-a+L/2) % L-L/2, (cen[:, 1]-b+L/2) % L-L/2); m |= dd <= R_
            res[f"{'+' if q > 0 else '-'}r{R_}"] = (round(float(s[m].mean()), 4), round(float((psi[f.TV[m]] > 0).all(1).mean()), 2))
    print(fn, "net", round(float(s.mean()), 5), "se", round(float(s.std()/np.sqrt(len(s))), 5), "lock", round(float((s*sig).mean()), 4), "sigsum", int(sig.sum()), res)
