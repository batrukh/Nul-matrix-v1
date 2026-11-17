"""(347) крок 3: світло на ґратці з довжинами ниток із постулату (δ у центрі). Котангенсна хвиля ü = −M⁻¹K u (інтринсивна, без розрізу),
плаский фронт у координатах розгортки Q (розріз позаду, +x). Кут між фронтами позаду = кут лінзи; очікування = δ."""
import sys, json, numpy as np
sys.path.insert(0, '/mnt/project-files/model/s52/r347'); from regge347 import *
def front_dir(Q, tarr, sel):
    m = sel & np.isfinite(tarr); X = np.c_[Q[m], np.ones(m.sum())]; coef, *_ = np.linalg.lstsq(X, tarr[m], rcond=None)
    return float(np.degrees(np.arctan2(coef[1], coef[0]))), float(1/np.hypot(*coef[:2]))
def run(deg, R=90, x0=-45.0, w=3.0, dt=0.2, T=150.0):
    P, Tr, E, inner = disk(R); n = len(P); c0 = int(np.argmin(np.hypot(*P.T)))
    d = np.zeros(n); d[c0] = np.radians(deg); u, err, it = solve(P, Tr, inner, d)
    Q = layout(P, Tr, u); Q = Q - Q[c0]
    K, M, area = cot_lap(P, Tr, u); rr = np.hypot(*P.T)
    x = Q[:, 0]; g = np.exp(-((x - x0)/w)**2); left = (x < -20) & (rr < R - 5)
    f = np.where(left, g, 0.0); v = np.where(left, 2*(x - x0)/w**2*g, 0.0)
    fix = rr >= R - 1.5; tarr = np.full(n, np.nan); fmax = np.zeros(n)
    acc = -(K @ f)/M; t = 0.0
    while t < T:
        v += 0.5*dt*acc; f += dt*v; f[fix] = 0; acc = -(K @ f)/M; v += 0.5*dt*acc; t += dt
        new = f > fmax; fmax = np.where(new, f, fmax); tarr = np.where(new & (f > 0.15), t, tarr)
    ang = np.degrees(np.arctan2(Q[:, 1], Q[:, 0])); r = np.hypot(*Q.T)
    lo = deg/2 + 8
    up = (r > 45) & (r < 80) & (ang > lo) & (ang < lo + 40); dn = (r > 45) & (r < 80) & (ang < -lo) & (ang > -lo - 40)
    a_up, v_up = front_dir(Q, tarr, up); a_dn, v_dn = front_dir(Q, tarr, dn)
    # кола: довжина кола радіуса r навколо апекса (r за розгорткою) = кутовий розмах × r
    a360 = np.mod(ang[(r > 20) & (r < 60)], 360); span = float(a360.max() - a360.min())
    return dict(deg=deg, err=float(err), it=int(it), dir_up=a_up, dir_dn=a_dn, lens=a_dn - a_up if False else abs(a_up - a_dn), v_up=v_up, v_dn=v_dn, span=span), Q, tarr, fmax, P, Tr
if __name__ == "__main__":
    res = {}
    for deg in [float(a) for a in sys.argv[1:]] or (0.0, 20.0, 40.0, 60.0):
        o, Q, tarr, fmax, P, Tr = run(deg); res[deg] = o; print(o, flush=True)
        np.savez(f'/mnt/project-files/model/s52/r347/lens_d{int(deg)}.npz', Q=Q, tarr=tarr, fmax=fmax, P=P)
    json.dump(res, open('/mnt/project-files/model/s52/r347/lens347.json', 'w'), indent=1)
