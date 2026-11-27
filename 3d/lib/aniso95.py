"""aniso95: № 64 / колізія (82)–(83) — чи бачить світло протополя анізотропію ниток (нахил τ).
Однорідний світ w64 (стріли чистого нахилу τ = i + 0,31 j, α = 1, запас 8, місцевий закон), живе протополе Φ★ (як local60 pf, (92)).
Нитки z, ψ і Φ★ рухаються разом; дві копії — без поштовху і з малим голдстоунівським поштовхом Φ★ у центрі.
Міряємо: (1) анізотропію ниток — інтенсивність трьох класів напрямків; (2) швидкість фронту хвилі Φ★ у 12 секторах кута.
Напрям нахилу в декартових координатах: (1 + 0,31/2, 0,31·√3/2) → ≈ 13°."""
import numpy as np, sys, json
import w64, phi90 as PH

def run(L=96, s0=8.0, ac=1.0, Tw=60.0, Tk=30.0, dt=0.005, pf=1.0, pg=0.2, amp=0.02, r0=2.0, seed=1, tilt=1):
    s, z, psi = w64.build(L, s0, ac, tilt=tilt, seed=seed); f = s.fab; c = w64.cls(f, L)
    chi_ = lambda p_: 0.5*(1 + np.tanh(p_/0.2)); dchi_ = lambda p_: (1 - np.tanh(p_/0.2)**2)/0.4
    P0 = np.diag(np.sort(np.linalg.eigvalsh(PH.vacuum(-1.0))))
    Phi = np.broadcast_to(P0, (L, L, 5, 5)).copy() + PH.sym0(0.01*np.random.default_rng(seed).standard_normal((L, L, 5, 5))); V = np.zeros_like(Phi)
    def step(z, psi, Phi, V):
        z, psi = s.step(z, psi, dt)
        aa = (1.0 - 2.0*chi_(psi)).reshape(L, L)
        V = V + dt*(PH.lap(Phi) - PH.dV(Phi, aa) - pg*V); Phi = PH.sym0(Phi + dt*V)
        t2 = np.trace(Phi @ Phi, axis1=-2, axis2=-1).reshape(-1)
        return z, psi + dt*pf*dchi_(psi)*t2, Phi, V
    t = 0.0; warm = []
    while t < Tw - 1e-9:
        z, psi, Phi, V = step(z, psi, Phi, V); t += dt
        if abs(t - round(t)) < dt/2 and round(t) % 10 == 0:
            I = np.abs(z)**2; warm.append(dict(t=round(t), fw=float((psi > 0).mean()), Ic=[float(I[c == k].mean()) for k in range(3)]))
            print(warm[-1], flush=True)
    # поштовх: поворот у зламаному напрямку (0,4) — голдстоунівська мода; без тертя Φ під час хвилі, щоб фронт не гас
    X, Y = PH.coords(L); cx, cy = X.mean(), Y.mean()
    best = None
    for p in (-1, 0, 1):
        for q in (-1, 0, 1):
            dx_, dy_ = X - cx + p*L + q*L/2, Y - cy + q*np.sqrt(3)/2*L; rr = np.hypot(dx_, dy_)
            best = (rr, dx_, dy_) if best is None else tuple(np.where(rr < best[0], v1, v0) for v0, v1 in zip(best, (rr, dx_, dy_)))
    r, dx, dy = best; ang = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
    G = np.zeros((5, 5)); G[0, 4], G[4, 0] = 1, -1; th = (amp*np.exp(-r**2/(2*r0**2)))[..., None, None]
    zA, pA, FA, VA = z.copy(), psi.copy(), Phi.copy(), V.copy()
    zB, pB, FB, VB = z.copy(), psi.copy(), Phi + th*(G @ Phi - Phi @ G), V.copy()
    pg0 = pg; pg = 0.0
    sec = (ang // 30).astype(int); rec = []; t = 0.0
    while t < Tk - 1e-9:
        zA, pA, FA, VA = step(zA, pA, FA, VA); zB, pB, FB, VB = step(zB, pB, FB, VB); t += dt
        if abs(t*2 - round(t*2)) < dt:
            e = np.sum((FB - FA)**2, axis=(-2, -1)) + np.sum((VB - VA)**2, axis=(-2, -1)); hot = e > 1e-3*e.max()
            fr = [float(r[hot & (sec == k)].max()) if (hot & (sec == k)).any() else 0.0 for k in range(12)]
            cen = [float((e*r)[sec == k].sum()/max(e[sec == k].sum(), 1e-30)) for k in range(12)]
            rec.append(dict(t=round(t, 2), front=fr, cen=cen, dpsi=float(np.abs(pB - pA).max()), dz=float(np.abs(zB - zA).max())))
    I = np.abs(zA)**2
    name = f"aniso95_L{L}_s{s0:g}_a{ac:g}_t{tilt}_pf{pf:g}_s{seed}"
    json.dump(dict(warm=warm, rec=rec, Ic=[float(I[c == k].mean()) for k in range(3)]), open(name + ".json", "w"))
    return name, warm, rec

if __name__ == "__main__":
    kw = {k: (int(v) if k in ("L", "seed", "tilt") else float(v)) for k, v in (a.split("=") for a in sys.argv[1:])}
    name, warm, rec = run(**kw); print(name)
    for q in rec[3::8]: print(q["t"], "front", [round(v) for v in q["front"]], "dpsi %.2e" % q["dpsi"], flush=True)
