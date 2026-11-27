"""light86: світло = інерція ниток (рішення автора 21.10.2026, журнал E; перевірка, не постулат).
Закон руху нитки: m·z̈ + ż = F(z, ψ) (m = 0 — стара модель ż = F); ψ — як раніше, першого порядку.
Тест: стан w64 (однорідний світ або однорідна Матриця, нахил стріл), прогрів з інерцією, далі дві копії — без поштовху і з малим
поштовхом фаз у плямі в центрі; різниця ниток |Δz| як функція відстані й часу. Хвиля — фронт іде з t лінійно (R ∝ t), дифузія — R ∝ √t.
mode=world: ψ у ямі світу, запас s0 (8 — стійкий однорідний світ, (84)); mode=matrix: ψ у ямі Матриці (запас 3,5)."""
import numpy as np, sys, json
import w64, local60 as M

def setup(L, s0, ac, mode, seed=1):
    s, z, psi = w64.build(L, s0, ac, tilt=1, seed=seed)
    if mode == "matrix":
        r_ = np.roots([1, 0, -1, -0.3]); psi = r_[np.isreal(r_)].real.min() + 0.01*np.random.default_rng(seed).standard_normal(len(psi))
    return s, z, psi

def step(s, z, v, psi, dt, m):
    if m == 0:
        z2, p2 = s.step(z, psi, dt); return z2, s.F(z2, p2)[0], p2
    def f(z_, v_, p_):
        Fz, Fp = s.F(z_, p_); return v_, (Fz - v_)/m, Fp
    k1 = f(z, v, psi); k2 = f(z + dt/2*k1[0], v + dt/2*k1[1], psi + dt/2*k1[2])
    k3 = f(z + dt/2*k2[0], v + dt/2*k2[1], psi + dt/2*k2[2]); k4 = f(z + dt*k3[0], v + dt*k3[1], psi + dt*k3[2])
    return (z + dt/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0]), v + dt/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1]), psi + dt/6*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2]))

def run(L=48, s0=8.0, ac=1.0, m=0.0, mode="world", Tw=40.0, Tk=16.0, dt=0.005, amp=0.01, r0=1.5, seed=1, kind="ph"):
    s, z, psi = setup(L, s0, ac, mode, seed); f = s.fab
    if m > 0: dt = min(dt, m/40)                         # явний RK4: крок менший за час інерції
    v = s.F(z, psi)[0]; t = 0.0; warm = []
    while t < Tw - 1e-9:
        z, v, psi = step(s, z, v, psi, dt, m); t += dt
        if abs(t - round(t)) < dt/2 and round(t) % 5 == 0: warm.append(dict(t=round(t), fw=float((psi > 0).mean()), I=float((np.abs(z)**2).mean())))
    # поштовх: фази ниток у плямі радіуса r0 біля центру (декартові координати середини нитки)
    X = f.xy[:, 0] + 0.5*f.xy[:, 1]; Y = np.sqrt(3)/2*f.xy[:, 1]
    dd = f.xy[f.head] - f.xy[f.tail]; dd = (dd + L//2) % L - L//2; mx = f.xy[f.tail] + dd/2
    ex = mx[:, 0] + 0.5*mx[:, 1]; ey = np.sqrt(3)/2*mx[:, 1]
    cx, cy = (L/2 + 0.5*L/2), np.sqrt(3)/2*L/2
    def tdist(x, y):                                     # відстань на торі (косокутна гратка)
        best = None
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                d_ = np.hypot(x - cx + a*L + b*L/2, y - cy + b*np.sqrt(3)/2*L)
                best = d_ if best is None else np.minimum(best, d_)
        return best
    TV = f.TV; tx = (X[TV[:, 0]] + X[TV[:, 1]] + X[TV[:, 2]])/3; ty = (Y[TV[:, 0]] + Y[TV[:, 1]] + Y[TV[:, 2]])/3
    rt = tdist(tx, ty)                                   # грубо (трикутники на шві тора дають хибний центр — їх мало, далеко від центру)
    re = tdist(ex, ey); kick = amp*np.exp(-re**2/(2*r0**2))
    zA, vA, pA = z.copy(), v.copy(), psi.copy(); kf = np.exp(1j*kick) if kind == "ph" else 1 + kick; zB = z*kf; vB = v*kf; pB = psi.copy()
    rec = []; t = 0.0; bins = np.arange(0, L/2 + 1, 1.0)
    while t < Tk - 1e-9:
        zA, vA, pA = step(s, zA, vA, pA, dt, m); zB, vB, pB = step(s, zB, vB, pB, dt, m); t += dt
        if abs(t*4 - round(t*4)) < dt*2:
            D2 = np.abs(zB - zA)**2
            dI = np.abs(np.abs(zB)**2 - np.abs(zA)**2)                     # калібрувально-інваріантні: інтенсивність ниток
            l1, l2, l3 = f.tri_links(zA); m1, m2, m3 = f.tri_links(zB)
            dh = np.abs(np.angle((m1*m2*m3)/(l1*l2*l3)))                    # і закрученість трикутників
            pI = np.histogram(re, bins, weights=dI)[0]/np.maximum(np.histogram(re, bins)[0], 1)
            ph_ = np.histogram(rt, bins, weights=dh)[0]/np.maximum(np.histogram(rt, bins)[0], 1)
            R = float(np.sqrt((D2*re**2).sum()/D2.sum()))
            prof = np.histogram(re, bins, weights=D2)[0]/np.maximum(np.histogram(re, bins)[0], 1)
            rec.append(dict(t=round(t, 2), R=R, tot=float(D2.sum()), rpk=float(bins[np.argmax(prof[2:]) + 2]), prof=[float(x) for x in prof], pI=[float(x) for x in pI], ph=[float(x) for x in ph_]))
    name = f"light86_{mode}{kind if kind != 'ph' else ''}_L{L}_s{s0:g}_a{ac:g}_m{m:g}" + (f"_dt{dt:g}" if dt != 0.005 else "")
    json.dump(dict(warm=warm, rec=rec, dt=dt), open(name + ".json", "w"))
    return name, warm, rec

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[1:]); kw = {k: (v if k in ("mode", "kind") else (int(v) if k in ("L", "seed") else float(v))) for k, v in kw.items()}
    name, warm, rec = run(**kw); print(name, "warm", warm[-1] if warm else None)
    for q in rec[::4]: print(q["t"], "R %.2f" % q["R"], "rpk", q["rpk"], "tot %.3g" % q["tot"], flush=True)
