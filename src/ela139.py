"""ela139: elastic substrate (author's idea of 26.10, hypothesis). A domain strains the substrate, and the substrate pulls back towards equilibrium (the state before the domain).
Strain e_i = Σ_j G_ij u_j, u = ψ − ψ_M (source — the domain), G = (−Δ + 1/ℓ²)⁻¹ on the triangular lattice: the strain spreads over the whole substrate and weakens with distance
(as ln r for r ≪ ℓ, ∝ e^(−r/ℓ) beyond). Energy (γ/2) Σ u·e, force on ψ: −γ e. ℓ = 0 means ℓ = ∞ (pure Poisson substrate, u with zero mean,
as in the Ohta–Kawasaki model). The rest is local60 unchanged; run: python3 ela139.py g=.. ell=.. <local60.run parameters>."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import local60 as M

def kernel(L, ell):
    k = 2*np.pi*np.fft.fftfreq(L); kx, ky = k[:, None], k[None, :]
    lam = 2*(3 - np.cos(kx) - np.cos(ky) - np.cos(ky - kx))       # −Δ of the triangular lattice (neighbours ±(1,0), ±(0,1), ±(−1,1)), node i = x·L + y
    if ell > 0: return 1.0/(lam + 1.0/ell**2)
    G = np.zeros_like(lam); G[lam > 1e-12] = 1.0/lam[lam > 1e-12]; return G

G_, ELL, PSIM = None, 8.0, None                    # reference model since 26.10 (author's decision): ℓ = 8, γ = 0.01; g=0 — model without elasticity
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
