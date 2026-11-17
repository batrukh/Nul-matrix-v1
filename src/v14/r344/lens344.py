"""(344) крок 2: світло повз дисклінацію. Хвиля ü = −(D − A)u на ґратці-конусі (5 секторів, r343) і пласкій (6). Розгортка: сектор s займає кути 30° + 60°s … 30° + 60°(s+1),
тож у конуса виріз ±30° довкола осі +x (позаду). Плаский фронт іде вздовж +x. Час приходу t(x) → напрямок фронту в кожному секторі (кут градієнта t).
Очікування 2D-ЗТВ (точкова маса = конус з вирізом δ = 60°): позаду два зображення, фронти під кутом ±δ/2 = ±30° один до одного (усього δ)."""
import sys, json, numpy as np, scipy.sparse as sp
sys.path.insert(0, '/mnt/project-files/model/s52/r343'); from cone343 import lattice
def run(nsec, R=120, x0=-45.0, w=3.0, dt=0.2, T=150.0):
    idx, E, ring = lattice(nsec, R); n = len(idx)
    P = np.zeros((n, 2)); th0 = np.radians(30.0) if nsec == 5 else 0.0
    for (s, a, b), i in idx.items():
        if (s, a, b) == (0, 0, 0): continue
        e1 = np.array([np.cos(th0 + np.pi/3*s), np.sin(th0 + np.pi/3*s)]); e2 = np.array([np.cos(th0 + np.pi/3*(s + 1)), np.sin(th0 + np.pi/3*(s + 1))]); P[i] = a*e1 + b*e2
    r_, c_ = [], []
    for i, j, t in E: r_ += [i, j]; c_ += [j, i]
    A = sp.csr_matrix((np.ones(len(r_)), (r_, c_)), shape=(n, n)); D = np.asarray(A.sum(1)).ravel(); Lap = sp.diags(D) - A
    c = np.sqrt(1.5); x = P[:, 0]; g = np.exp(-((x - x0)/w)**2); left = (x < -20) & (ring < R - 5)
    u = np.where(left, g, 0.0); v = np.where(left, 2*(x - x0)/w**2*g*c, 0.0)   # u = g(x − x0 − ct)
    fix = ring >= R - 1; tarr = np.full(n, np.nan); umax = np.zeros(n); snaps = {}
    t = 0.0; acc = -(Lap @ u); k = 0
    while t < T:
        v += 0.5*dt*acc; u += dt*v; u[fix] = 0; acc = -(Lap @ u); v += 0.5*dt*acc; t += dt; k += 1
        new = u > umax; umax = np.where(new, u, umax); tarr = np.where(new & (u > 0.15), t, tarr)
        for ts in (40.0, 80.0, 120.0):
            if abs(t - ts) < dt/2: snaps[ts] = u.copy()
    return P, ring, tarr, snaps, umax
def front_dir(P, ring, tarr, sel):
    m = sel & np.isfinite(tarr)
    X = np.c_[P[m], np.ones(m.sum())]; coef, *_ = np.linalg.lstsq(X, tarr[m], rcond=None)
    gx, gy = coef[:2]; return float(np.degrees(np.arctan2(gy, gx))), float(1/np.hypot(gx, gy))
if __name__ == "__main__":
    res = {}
    for nsec in (6, 5):
        P, ring, tarr, snaps, umax = run(nsec)
        ang = np.degrees(np.arctan2(P[:, 1], P[:, 0])); r = np.hypot(*P.T)
        up = (r > 50) & (r < 85) & (ang > 35) & (ang < 75)       # сектор позаду-вгорі
        dn = (r > 50) & (r < 85) & (ang < -35) & (ang > -75)
        a_up, v_up = front_dir(P, ring, tarr, up); a_dn, v_dn = front_dir(P, ring, tarr, dn)
        res[nsec] = dict(dir_up=a_up, dir_dn=a_dn, v_up=v_up, v_dn=v_dn, c=float(np.sqrt(1.5)))
        print(nsec, res[nsec], flush=True)
        np.savez(f'lens_n{nsec}.npz', P=P, ring=ring, tarr=tarr, umax=umax, **{f's{int(k)}': v for k, v in snaps.items()})
    json.dump(res, open('lens344.json', 'w'), indent=1)
