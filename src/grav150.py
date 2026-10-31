"""grav150: candidate postulate (150), author's idea of 27.10 — coupling of the Primary field (links) and the protofield Φ★.
The links of Φ★ and of the gauge field lie on the same links as the Matrix links; the stiffness of a link ∝ its intensity:
    H = Σ_l (1 + s_l)·(1/3)|U Φ' Uᵀ − Φ|² + Σ_tri (1 + s_tri)·kb·(5 − tr H) + [rest of gauge91 unchanged]
        + (K/2)·Σ [ |∇s|² + s²/ℓ² ]          (s = (A − A₀)/A₀ — relative change of intensity; elastic substrate (139) as the strain energy)
The strain adjusts fast (adiabatically): s = −G·ρ/K, G = (−Δ + 1/ℓ²)⁻¹, ρ — density of the "gradient" energy of Φ★ at the nodes
(links + triangles). ℓ = 0 means ℓ = ∞ (Poisson, zero mean). Effectively: H_eff = H₀ − (1/2K)·ρ·G·ρ.
Kinetic terms (Π, E) and the potential V are not multiplied: wave speed c² ∝ (1 + s)."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import gauge91 as G
from gauge91 import sh, put, T_, PA, tr, DIRS

def kernel(L, ell):
    k = 2*np.pi*np.fft.fftfreq(L); kx, ky = k[:, None], k[None, :]
    lam = 2*(3 - np.cos(kx) - np.cos(ky) - np.cos(kx - ky))       # −Δ of the triangular lattice with neighbours ±(1,0), ±(0,1), ±(−1,1)
    if ell > 0: return 1.0/(lam + 1.0/ell**2)
    Gk = np.zeros_like(lam); Gk[lam > 1e-12] = 1.0/lam[lam > 1e-12]; return Gk

class GaugeA(G.Gauge):
    def __init__(s, L, K, ell, ka=1.0, kb=2.0, a=-1.0, couple=True):
        super().__init__(np.full((L, L), a), np.ones((3, L, L)), np.ones((2, L, L)), ka, kb)
        s.L = L; s.K = K; s.ell = ell; s.Gk = kernel(L, ell); s.couple = couple
        s.sn = np.zeros((L, L)); s.fl = np.ones((3, L, L)); s.wt0 = np.ones((2, L, L))
    def grad_parts(s, P, U):
        el = np.array([(1/3)*np.sum((U[m] @ sh(P, d) @ T_(U[m]) - P)**2, axis=(-2, -1)) for m, d in enumerate(DIRS)])
        Hu, Hd = s.hol(U); ep = np.array([s.kb*(5 - tr(Hu)), s.kb*(5 - tr(Hd))])
        return el, ep
    def rho(s, P, U):
        el, ep = s.grad_parts(P, U); r = np.zeros((s.L, s.L))
        for m, d in enumerate(DIRS): r += 0.5*el[m] + 0.5*put(el[m], d)
        r += (ep[0] + put(ep[0], (1, 0)) + put(ep[0], (0, 1)))/3
        r += (put(ep[1], (1, 0)) + put(ep[1], (1, 1)) + put(ep[1], (0, 1)))/3
        return r
    def update_s(s, P, U, frozen=None):
        if frozen is not None: s.sn = frozen
        elif s.couple:
            r = s.rho(P, U)
            if s.ell <= 0: r = r - r.mean()
            s.sn = -np.real(np.fft.ifft2(np.fft.fft2(r)*s.Gk))/s.K
        f = 1.0 + s.sn
        s.fl = np.array([0.5*(f + sh(f, d)) for d in DIRS])
        s.wt = np.array([(f + sh(f, DIRS[0]) + sh(f, DIRS[1]))/3, (sh(f, DIRS[0]) + sh(f, (1, 1)) + sh(f, DIRS[1]))/3])
    def forces(s, P, U):
        FP = -G.dV(P, s.a, s.b)
        R = [np.zeros_like(U[0]) for _ in range(3)]
        for m, d in enumerate(DIRS):
            X = U[m] @ sh(P, d) @ T_(U[m]); D = X - P; w = s.fl[m][..., None, None]
            FP += w*(2/3)*D
            FP -= put(w*(2/3)*(T_(U[m]) @ D @ U[m]), d)
            R[m] += w*(4/3)*(D @ X)
        U0, U1, U2 = U
        bu = (s.kb*s.wt[0])[..., None, None]; bd = (s.kb*s.wt[1])[..., None, None]
        A, B = U0, sh(U2, DIRS[0]); Hu = A @ B @ T_(U1)
        R[0] += -bu*T_(Hu); R[2] += put(-bu*(T_(A) @ T_(Hu) @ A), DIRS[0]); R[1] += -bu*Hu
        A, B, C = sh(U1, DIRS[0]), sh(U0, DIRS[1]), sh(U2, DIRS[0]); Hd = A @ T_(B) @ T_(C)
        R[1] += put(-bd*T_(Hd), DIRS[0]); R[0] += put(-bd*(T_(C) @ Hd @ C), DIRS[1]); R[2] += put(-bd*Hd, DIRS[0])
        FE = np.array([-(2/(s.ka*s.wl[m]))[..., None, None]*PA(R[m]) for m in range(3)])
        return G.sym0(FP), FE
    def edens(s, P, Pi, U, E):
        e = 0.5*np.sum(Pi**2, axis=(-2, -1))
        P2 = P @ P; t2 = tr(P2)
        e += s.a/2*t2 - s.b/3*tr(P2 @ P) + t2**2/4 + 0.5/4*tr(P2 @ P2)
        for m, d in enumerate(DIRS):
            D = U[m] @ sh(P, d) @ T_(U[m]) - P
            e += s.fl[m]*(1/3)*np.sum(D**2, axis=(-2, -1)) + s.ka*s.wl[m]/4*np.sum(E[m]**2, axis=(-2, -1))
        Hu, Hd = s.hol(U)
        e += s.kb*(s.wt[0]*(5 - tr(Hu)) + s.wt[1]*(5 - tr(Hd)))
        return e
    def step(s, P, Pi, U, E, dt, gam=0.0, frozen=None):
        s.update_s(P, U, frozen)
        return super().step(P, Pi, U, E, dt, gam)
    def energy(s, P, Pi, U, E):
        """H_eff = H₀ (without factors) + Σ s·ρ/2 for adiabatic s; returns (H_eff, H₀)."""
        sv = s.sn; s.sn = np.zeros_like(sv); s.update_s(P, U, frozen=s.sn); H0 = s.edens(P, Pi, U, E).sum(); s.sn = sv; s.update_s(P, U, frozen=sv)
        return H0 + 0.5*float((s.sn*s.rho(P, U)).sum()), H0
