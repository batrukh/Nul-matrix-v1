"""photon100: речовина, крок 3б — «фотон» (хвильовий пакет незламаного напрямку O(2) калібрувального поля) налітає на Z₂-вихор (99).
Старт: усталений стан gvort99 d = 32 (вихори при X ≈ 55,5 і 86,5, Y ≈ 41). Пакет біжить уздовж +X від X0 = 71 до правого вихору,
прицільна відстань b (по Y). Поперечна поляризація: ланки μ = 1, 2 (мають Y-складову √3/2). Генератор O(2) у вузлі: G = B·J·Bᵀ,
B — площина власних значень v, орієнтація узгоджена з центром пакета. Контроль: той самий пакет у вакуумі без вихорів.
Дві копії (з пакетом і без) — енергія різниці: куди пішла хвиля (пройшла / розсіялась / назад), віддача вихору, потік Z₂ після проходу."""
import numpy as np, json, sys
import phi90 as PH, gauge91 as G, matter98 as M, gvort99 as GV

J = np.array([[0.0, -1.0], [1.0, 0.0]])

def packet(P, U, X, Y, X0, Y0, amp=0.15, w=3.0, wy=4.0, k=1.0):
    B = M.planes(P); i0 = np.unravel_index(np.argmin((X - X0)**2 + (Y - Y0)**2), X.shape); Br = B[i0]
    flip = np.linalg.det(np.einsum("ia,...ib->...ab", Br, B)) < 0; B = B.copy(); B[flip, :, 1] *= -1
    Gk = B @ J @ np.swapaxes(B, -1, -2)
    env = np.exp(-(X - X0)**2/(2*w**2) - (Y - Y0)**2/(2*wy**2)); f = amp*env*np.cos(k*(X - X0))
    fp = amp*env*(-(X - X0)/w**2*np.cos(k*(X - X0)) - k*np.sin(k*(X - X0)))
    U = U.copy(); E = np.zeros_like(U); s = np.sqrt(3)/2
    for m in (1, 2):
        U[m] = G.cayley((s*f)[..., None, None]*Gk) @ U[m]; E[m] = (-s*fp)[..., None, None]*Gk
    return U, E

def run(src="gvort99_L96_d32_ka1_kb2_Tr60_g0.npz", b=0.0, T=40.0, dt=0.05, vac=0, ka=1.0, kb=2.0, X0=71.0):
    z = np.load(src); P, U = z["P"], z["U"]; L = P.shape[0]
    if vac: P = np.broadcast_to(np.diag([M.U_]*3 + [M.V_]*2), P.shape).copy(); U = np.broadcast_to(np.eye(5), U.shape).copy()
    X, Y = PH.coords(L); Y0 = 41.0 + b; xv = 86.5
    g = G.Gauge(np.full((L, L), -1.0), np.ones((3, L, L)), np.ones((2, L, L)), ka, kb)
    UB, EB = packet(P, U, X, Y, X0, Y0)
    A = [P.copy(), np.zeros_like(P), U.copy(), np.zeros_like(U)]; Bs = [P.copy(), np.zeros_like(P), UB, EB]
    t = 0.0; rec = []
    while t < T - 1e-9:
        A = list(g.step(*A, dt)); Bs = list(g.step(*Bs, dt)); t += dt
        if abs(t - round(t)) < dt/2:
            e = g.edens(*Bs) - g.edens(*A); ep = np.clip(e, 0, None); tot = ep.sum()
            fwd = ep[X > xv + 6].sum()/tot; back = ep[X < X0 - 8].sum()/tot
            side = ep[(X >= X0 - 8) & (X <= xv + 6) & (np.abs(Y - Y0) > 10)].sum()/tot
            cx = float((ep*X).sum()/tot)
            j0 = L//2; i2 = int(round(xv - 0.5*j0))
            wl = round(GV.wilson(Bs[0], Bs[2], i2, j0, 8, L), 3) if not vac else 0.0
            rec.append(dict(t=round(t, 1), tot=float(e.sum()), cx=cx, fwd=float(fwd), back=float(back), side=float(side), W=wl))
            print(rec[-1], flush=True)
    # віддача: зсув центроїда енергії правого вихору (копія з пакетом мінус без)
    def cen(S):
        e0 = g.edens(S[0], 0*S[1], S[2], 0*S[3]); m = (np.abs(X - xv) < 10) & (np.abs(Y - 41) < 10); w_ = np.clip(e0, 0, None)*m
        return float((w_*X).sum()/w_.sum()), float((w_*Y).sum()/w_.sum())
    rc = (cen(Bs), cen(A)) if not vac else None
    name = f"photon100_b{b:g}_vac{vac}"; json.dump(dict(rec=rec, recoil=rc), open(name + ".json", "w"))
    print("recoil", rc); return name

if __name__ == "__main__":
    kw = {k: (v if k == "src" else (int(v) if k == "vac" else float(v))) for k, v in (a.split("=") for a in sys.argv[1:])}
    run(**kw)
