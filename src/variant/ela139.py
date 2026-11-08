"""ela139: пружне полотно (ідея автора 26.10, гіпотеза). Світ розтягує полотно, полотно тягне назад до рівноваги (стану до світу).
Розтяг e_i = Σ_j G_ij u_j, u = ψ − ψ_М (джерело — світ), G = (−Δ + 1/ℓ²)⁻¹ на трикутній ґратці: розтяг іде всім полотном і слабшає з відстанню
(на r ≪ ℓ як ln r, далі ∝ e^(−r/ℓ)). Енергія (γ/2) Σ u·e, сила на ψ: −γ e. ℓ = 0 означає ℓ = ∞ (чисте полотно Пуассона, u без середнього,
як у моделі Огти–Кавасакі). Решта — local60 без змін; запуск: python3 ela139.py g=.. ell=.. <параметри local60.run>."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import local60 as M

def kernel(L, ell):
    k = 2*np.pi*np.fft.fftfreq(L); kx, ky = k[:, None], k[None, :]
    lam = 2*(3 - np.cos(kx) - np.cos(ky) - np.cos(ky - kx))       # −Δ трикутної ґратки (сусіди ±(1,0), ±(0,1), ±(−1,1)), вузол i = x·L + y
    if ell > 0: return 1.0/(lam + 1.0/ell**2)
    G = np.zeros_like(lam); G[lam > 1e-12] = 1.0/lam[lam > 1e-12]; return G

G_, ELL, PSIM = None, 8.0, None                    # базова модель з 26.10 (рішення автора, журнал E): ℓ = 8, γ = 0,01; g=0 — модель без пружності
base_F = M.Loc.F
def F(s, z, psi, parts=False):
    out = base_F(s, z, psi, parts)
    if parts or not getattr(s, "gam", 0): return out
    L = s.fab.L if hasattr(s.fab, "L") else int(round(np.sqrt(len(psi))))
    u = (psi - s.psiM).reshape(L, L)
    if s.ell <= 0: u = u - u.mean()
    e = np.real(np.fft.ifft2(np.fft.fft2(u)*s.G))
    s.e_last = e
    return out[0], out[1] - s.gam*e.ravel()
M.Loc.F = F

orig_init = M.Loc.__init__
def init(s, *a, **k):
    orig_init(s, *a, **k); L = k.get("L"); s.gam = GAM; s.ell = ELL; s.G = kernel(L, ELL)
    r_ = np.roots([1, 0, -1, -k.get("h", 0.3)]); s.psiM = r_[np.isreal(r_)].real.min()
M.Loc.__init__ = init

if __name__ == "__main__":
    kw = {}; GAM = 0.01
    for a in sys.argv[1:]:
        k, v = a.split("=")
        if k == "g": GAM = float(v)
        elif k == "ell": ELL = float(v)
        else: kw[k] = int(v) if k in ("L", "n", "seed", "sk", "ln", "snp") else float(v)
    print(M.run(**kw))
