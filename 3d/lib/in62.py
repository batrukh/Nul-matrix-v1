"""in62: № 62 крок 3 — чи стан УСЕРЕДИНІ світів анізотропний відносно осі світу.
Для ниток з обома кінцями у світі (ψ>0) і поза світами: середня інтенсивність |z|² за трьома класами напрямків
(декартові кути 0°, 60°, 120°). Анізотропія A = (max − min)/середнє. Контроль — стани без нахилу (66, iso) і поза світами."""
import numpy as np, glob
import aniso62 as A
f = A.f; L = A.L
d = f.xy[f.head] - f.xy[f.tail]; d = (d + L//2) % L - L//2
cls = np.full(f.m, -1)
cls[(np.abs(d[:, 0]) == 1) & (d[:, 1] == 0)] = 0; cls[(d[:, 0] == 0) & (np.abs(d[:, 1]) == 1)] = 1; cls[(d[:, 0] == -d[:, 1]) & (d[:, 0] != 0)] = 2
assert (cls >= 0).all()
ANG = np.array([0., 60., 120.])
def an(fn):
    s = np.load(fn); z = s["z"]; psi = s["psi"]; I = np.abs(z)**2
    inw = (psi[f.tail] > 0) & (psi[f.head] > 0); out = (psi[f.tail] <= 0) & (psi[f.head] <= 0)
    W = A.worlds(psi, mn=30); axis = None
    if W:
        wts = np.array([x[0] for x in W], float); a2 = np.radians(2*np.array([x[1] for x in W]))
        axis = np.degrees(np.arctan2((wts*np.sin(a2)).sum(), (wts*np.cos(a2)).sum()))/2 % 180
    res = {}
    for lab, msk in (("у світах", inw), ("поза", out)):
        if msk.sum() < 30: res[lab] = None; continue
        m = np.array([I[msk & (cls == c)].mean() for c in range(3)]); se = np.array([I[msk & (cls == c)].std()/np.sqrt((msk & (cls == c)).sum()) for c in range(3)])
        res[lab] = (np.round(m, 2), np.round(se, 2), round(float((m.max() - m.min())/m.mean()), 3), int(msk.sum()))
    return axis, res
groups = {"нахил iso (79)": sorted(glob.glob("lc_s03.5_n100_L96_s1_tau8_D1_g0.02_ga*_iso.npz")) + ["lc_s03.5_n100_L96_s1_tau8_D1_g0.02_iso.npz", "lc_s03.5_n100_L96_s2_tau8_D1_g0.02_iso.npz"],
          "нахил старі (68)–(75)": [x for x in A.groups["нахил (загоєний)"]],
          "без нахилу (66) iso": [f"lc_s03.5_n100_L96_s{s}_tau8_iso.npz" for s in (1, 2, 3, 4)]}
for g, fs in groups.items():
    print("==", g)
    for fn in fs:
        ax, r = an(fn)
        print("  ", fn.split("/")[-1], "вісь", None if ax is None else round(ax, 1))
        for k, v in r.items():
            if v: print("      %-8s I(0°,60°,120°) = %s ± %s  A = %.3f  ниток %d" % (k, v[0], v[1], v[2], v[3]))
