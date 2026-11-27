"""real102: речовина, крок 5 — Z₂-вихор (99) у справжньому світі local60 (L144, реалізація τ 2, світ 864 вузли, глибина 8).
ψ заморожене зі стану local60 (як у (91)); a = 1 − 2χ(ψ): у світі протополе розщеплене, у Матриці ні.
Калібрувальне поле: варіант А — «спить» у Матриці (вага w = max(χ, floor)), варіант Б — усюди (w = 1).
Дві копії: без вихору і з вихором у найглибшій точці світу (φ = θ/2 у площині (e3, e4), ланки — згорнута різниця φ).
Тертя до t = Tr, далі вільна динаміка. Міряємо: потік Z₂ (петлі Вільсона r = 3, 5, 7 навколо стартової точки),
енергію вихору (різниця копій), центроїд енергії вихору, частку енергії вихору в Матриці (ψ < 0)."""
import numpy as np, json, sys
import phi90 as PH, gauge91 as G, matter98 as M, gvort99 as GV

def run(src="lc_s03.5_n100_L144_s2_tau8_D1_g0.02_iso.npz", var="world", floor=0.02, Tr=40.0, T=100.0, dt=0.05, ka=1.0, kb=2.0, ci=78, cj=94):
    psi = np.load(src)["psi"]; L = int(round(np.sqrt(psi.size))); psi = psi.reshape(L, L)
    chi = 0.5*(1 + np.tanh(psi/0.2)); a = 1.0 - 2.0*chi
    wl, wt = G.weights(chi, floor) if var == "world" else (np.ones((3, L, L)), np.ones((2, L, L)))
    g = G.Gauge(a, wl, wt, ka, kb)
    X, Y = PH.coords(L); cx, cy = X[ci, cj], Y[ci, cj]
    P0 = np.diag([M.U_]*3 + [M.V_]*2)*chi[..., None, None]
    rng = np.random.default_rng(1); noise = PH.sym0(0.01*rng.standard_normal(P0.shape))
    PA = PH.sym0(P0 + noise); UA = np.broadcast_to(np.eye(5), (3, L, L, 5, 5)).copy()
    # вихор: кут навколо (cx, cy) у найближчому образі тора
    _, dx, dy = M.wrapd(X, Y, cx, cy, L); phi = np.arctan2(dy, dx)/2
    R = M.rot(phi, 2, 3); PB = PH.sym0(R @ P0 @ np.swapaxes(R, -1, -2) + noise)
    UB = np.array([M.rot(GV.wrap_half(phi - np.roll(phi, (-d[0], -d[1]), axis=(0, 1))), 2, 3) for d in G.DIRS])
    for m in range(3):
        Xn = rng.standard_normal(UB[m].shape)*0.02; Xn = Xn - np.swapaxes(Xn, -1, -2); UB[m] = G.cayley(Xn) @ UB[m]
    SA = [PA, np.zeros_like(PA), UA, np.zeros_like(UA)]; SB = [PB, np.zeros_like(PB), UB, np.zeros_like(UB)]
    inM = psi < 0; t = 0.0; rec = []
    while t < Tr + T - 1e-9:
        gg = 1.0 if t < Tr else 0.0
        SA = list(g.step(*SA, dt, gam=gg)); SB = list(g.step(*SB, dt, gam=gg)); t += dt
        if abs(t - round(t)) < dt/2 and round(t) % 4 == 0:
            e = g.edens(*SB) - g.edens(*SA); ep = np.clip(e, 0, None)
            W = [round(GV.wilson(SB[0], SB[2], ci, cj, r, L), 3) for r in (3, 5, 7)]
            r_ = M.wrapd(X, Y, cx, cy, L)[0]; m_ = r_ < 12
            cen = (float((ep*X)[m_].sum()/max(ep[m_].sum(), 1e-30)), float((ep*Y)[m_].sum()/max(ep[m_].sum(), 1e-30)))
            rec.append(dict(t=round(t, 1), E=float(e.sum()), near=float(ep[r_ < 8].sum()/max(ep.sum(), 1e-30)),
                            inM=float(ep[inM].sum()/max(ep.sum(), 1e-30)), W=W, cen=cen, chi_c=float(chi[ci, cj])))
            print(rec[-1], flush=True)
    name = f"real102_{src.split("_")[3]}_{src.split("_")[-1][:-4]}_{var}_f{floor:g}"; json.dump(dict(rec=rec, c=(float(cx), float(cy))), open(name + ".json", "w"))
    np.savez(name + ".npz", PB=SB[0], UB=SB[2], PA=SA[0], UA=SA[2], psi=psi)
    return name

if __name__ == "__main__":
    kw = {k: (v if k in ("src", "var") else (int(v) if k in ("ci", "cj") else float(v))) for k, v in (a.split("=") for a in sys.argv[1:])}
    run(**kw)
