"""grav153: метричний зв'язок (152) з власним джерелом. Натяг s — пружне полотно (139): H = H_m(N = 1 + s) + (K/2)·Σ[|∇s|² + s²/ℓ²],
s підлаштовується швидко (адіабатично): K(−Δ + 1/ℓ²)s = −ρ, ρ = ∂H_m/∂N у вузлі — джерело випливає з тієї ж енергії, вільного вибору немає.
Потенціал рахується від вакууму (V − V₀): вакуум не має енергії і не є джерелом (космологічний член = 0 — вибір).
ρ (q = 1, a = 1/N): 3N²·½Π² (кін. Φ★) − (V − V₀)/N² + Σ_l ½·D-член + Σ_l ½·E-енергія/N_l + Σ_tri ⅓·3N_t²·kb(5 − tr H).
Для статичного солітона при N = 1: ρ = град + 3·B − V, а за Деріком (V = B) ∫ρ = ∫(град + B + V) = m — уся маса."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../r152")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../r150")
import gauge91 as G, grav152 as GM, matter98 as M
from gauge91 import sh, put, T_, tr, DIRS
from grav150 import kernel

P0 = np.diag([M.U_]*3 + [M.V_]*2)
def Vf(P, a, b):
    P2 = P @ P; t2 = tr(P2); return a/2*t2 - b/3*tr(P2 @ P) + t2**2/4 + 0.5/4*tr(P2 @ P2)

class GaugeMS(GM.GaugeM):
    def __init__(s, L, K, ell, q=1.0, ka=1.0, kb=2.0, a=-1.0, couple=True):
        s.L, s.K, s.ell, s.q, s.couple = L, K, ell, q, couple
        s.Gk = kernel(L, ell); s.sn = np.zeros((L, L)); s.kb_, s.ka_, s.a_ = kb, ka, a
        s.V0 = float(Vf(P0, a, 0.0)); s.set_N(np.ones((L, L)))
    def set_N(s, Nn):
        GM.GaugeM.__init__(s, Nn, q=s.q, ka=s.ka_, kb=s.kb_, a=s.a_)
        s.Nl = s.fl; s.Nt = s.wt**(1/(1 + 2*s.q))
    def edens(s, P, Pi, U, E):
        return GM.GaugeM.edens(s, P, Pi, U, E) - s.cV*s.V0
    def rho(s, P, Pi, U, E):
        q, Nn = s.q, s.Nn
        r = (2*q + 1)*Nn**(2*q)*0.5*np.sum(Pi**2, axis=(-2, -1)) + (1 - 2*q)*Nn**(-2*q)*(Vf(P, s.a, s.b) - s.V0)
        for m, d in enumerate(DIRS):
            D = U[m] @ sh(P, d) @ T_(U[m]) - P
            x = (1/3)*np.sum(D**2, axis=(-2, -1)) + s.ka*s.wl[m]/4*np.sum(E[m]**2, axis=(-2, -1))/s.Nl[m]
            r += 0.5*x + 0.5*put(x, d)
        Hu, Hd = s.hol(U); c = (1 + 2*q)*s.kb
        xu = c*s.Nt[0]**(2*q)*(5 - tr(Hu)); xd = c*s.Nt[1]**(2*q)*(5 - tr(Hd))
        r += (xu + put(xu, DIRS[0]) + put(xu, DIRS[1]))/3
        r += (put(xd, DIRS[0]) + put(xd, (1, 1)) + put(xd, DIRS[1]))/3
        return r
    def update_s(s, P, Pi, U, E, frozen=None):
        if frozen is not None: s.sn = frozen
        elif s.couple:
            r = s.rho(P, Pi, U, E)
            if s.ell <= 0: r = r - r.mean()
            s.sn = -np.real(np.fft.ifft2(np.fft.fft2(r)*s.Gk))/s.K
        s.set_N(1.0 + s.sn)
    def settle(s, P, Pi, U, E, frozen=None, it=3):
        """Самоузгоджене s (ітерації); E — швидкість, тож при зміні N зберігаємо канонічний імпульс ланки ∝ wl·E. Повертає новe E."""
        wl0 = s.wl.copy()
        for _ in range(it if frozen is None else 1): s.update_s(P, Pi, U, E*(wl0/s.wl)[..., None, None], frozen)
        return E*(wl0/s.wl)[..., None, None]
    def step(s, P, Pi, U, E, dt, gam=0.0, frozen=None, it=3):
        E = s.settle(P, Pi, U, E, frozen, it)
        return GM.GaugeM.step(s, P, Pi, U, E, dt, gam)
    def energy(s, P, Pi, U, E):
        """H_eff = H_m(N) + полотно; полотно = (K/2)·s·(−Δ+1/ℓ²)·s, рахуємо через FFT."""
        Hm = s.edens(P, Pi, U, E).sum(); sk = np.fft.fft2(s.sn)
        lam = np.where(s.Gk > 0, 1.0/np.where(s.Gk > 0, s.Gk, 1), 0.0)
        return Hm + 0.5*s.K*float(np.real(np.sum(np.abs(sk)**2*lam))/s.L**2), Hm
