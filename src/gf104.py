"""gf104: протополе Φ★ + калібрувальне поле SO(5) (gauge91) усередині повної моделі local60 (живі нитки, живе ψ).
Варіант А: калібрувальне поле «спить» у Матриці (ваги max(χ(ψ), floor)). Крок раз на K кроків ниток (dt_g = K·dt), тертя pg.
Зворотна дія на світ та сама, що в (92): ψ̇ += pf·χ'(ψ)·trΦ².
Детектор Z₂-дефекту калібрувально-інваріантний: голономія площини v по трикутнику з перенесенням ланками U, det < 0.
Посадка вихору (plant): сингулярне калібрувальне перетворення g = R(φ), φ = θ/2 навколо точки в площині (e_u, e_v) з власних векторів Φ★ у центрі,
плюс R(π) на ланках, що перетинають розріз (згорнута різниця кутів), — так розріз зникає, а потік Z₂ лишається в центрі."""
import numpy as np
import phi90 as PH, gauge91 as G, matter98 as M, gvort99 as GV

chi_ = lambda p: 0.5*(1 + np.tanh(p/0.2)); dchi_ = lambda p: (1 - np.tanh(p/0.2)**2)/0.4

class GF:
    def __init__(s, L, seed, floor=0.02, ka=1.0, kb=2.0, pg=0.2, b=0.0):
        s.b = b; s.L = L; s.floor = floor; s.ka = ka; s.kb = kb; s.pg = pg
        s.P = PH.sym0(0.01*np.random.default_rng(700 + seed).standard_normal((L, L, 5, 5))); s.Pi = np.zeros_like(s.P)
        s.U = np.broadcast_to(np.eye(5), (3, L, L, 5, 5)).copy(); s.E = np.zeros_like(s.U)

    def step(s, psi, dt, pf):
        p = psi.reshape(s.L, s.L); chi = chi_(p); wl, wt = G.weights(chi, s.floor)
        g = G.Gauge(1.0 - 2.0*chi, wl, wt, s.ka, s.kb, s.b)
        s.P, s.Pi, s.U, s.E = g.step(s.P, s.Pi, s.U, s.E, dt, gam=s.pg)
        t2 = np.trace(s.P @ s.P, axis1=-2, axis2=-1)
        return psi + dt*pf*(dchi_(p)*t2).reshape(-1)

    def defects(s, psi, thr=0.3, aware=False):
        """det голономії площини по трикутниках (2, L, L) і маска світу (усі три вершини ψ > thr).
        aware (120): вакууми Φ★ і −Φ★ рівноправні (b = 0); у домені з трьома додатними власними значеннями площина v —
        подвійна пара від'ємних, а не два найбільші з трьох майже рівних (старий детектор давав там хибні вихори)."""
        if aware:
            w, v = np.linalg.eigh(s.P); B = np.where(((w > 0).sum(-1) >= 3)[..., None, None], v[..., :, :2], v[..., :, 3:])
        else: B = M.planes(s.P)
        U0, U1, U2 = s.U; sh = G.sh; T = lambda A: np.swapaxes(A, -1, -2)
        ov = lambda Bi, Uij, Bj: M.polar2(T(Bi) @ Uij @ Bj)
        Ba, Bb, Bc, Bd = B, sh(B, (1, 0)), sh(B, (0, 1)), sh(B, (1, 1))
        Hu = ov(Ba, U0, Bb) @ ov(Bb, sh(U2, (1, 0)), Bc) @ ov(Bc, T(U1), Ba)
        Hd = ov(Bb, sh(U1, (1, 0)), Bd) @ ov(Bd, T(sh(U0, (0, 1))), Bc) @ ov(Bc, T(sh(U2, (1, 0))), Bb)
        det = np.array([np.linalg.det(Hu), np.linalg.det(Hd)])
        p = psi.reshape(s.L, s.L); q = lambda d: sh(p, d) > thr
        w = np.array([q((0, 0)) & q((1, 0)) & q((0, 1)), q((1, 0)) & q((1, 1)) & q((0, 1))])
        return det, w

    def plant(s, i0, j0):
        L = s.L; X, Y = PH.coords(L); _, dx, dy = M.wrapd(X, Y, X[i0, j0], Y[i0, j0], L); phi = np.arctan2(dy, dx)/2
        w, v = np.linalg.eigh(s.P[i0, j0]); eu, ev = v[:, 2], v[:, 3]          # один u- і один v-вектор
        Gm = np.outer(ev, eu) - np.outer(eu, ev)
        def R(a):
            c, sn = np.cos(a)[..., None, None], np.sin(a)[..., None, None]
            Pp = np.outer(eu, eu) + np.outer(ev, ev)
            return np.eye(5) - Pp + c*Pp + sn*Gm
        g = R(phi); s.P = PH.sym0(g @ s.P @ np.swapaxes(g, -1, -2)); s.Pi = g @ s.Pi @ np.swapaxes(g, -1, -2)
        for m, d in enumerate(G.DIRS):
            pn = G.sh(phi, d); dlt = phi - pn; c = GV.wrap_half(dlt) - dlt          # 0 або ±π на розрізі
            gn = G.sh(g, d)
            s.U[m] = g @ s.U[m] @ np.swapaxes(gn, -1, -2) @ R(c); s.E[m] = g @ s.E[m] @ np.swapaxes(g, -1, -2)
        return float(X[i0, j0]), float(Y[i0, j0])
