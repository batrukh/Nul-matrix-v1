"""grav152: «метричний» зв'язок (151), прочитання «натяг × довжина»: поля Φ★ і калібрувальне живуть у метриці
    ds² = −N²dt² + a²(dx² + dy²),  N = 1 + s (час з натягу, (25)),  a = N^(−q) (q = 1: збереження A·ℓ → γ = 1).
Гамільтоніан (2D, на вузол / ланку / трикутник):
    Σ_n [ (N/a²)·½|Π|² + N·a²·V(Φ) ] + Σ_l [ N_l·(1/3)|UΦ'Uᵀ − Φ|² + (ka/(4N_l))|E|² ] + Σ_tri (N_t/a_t²)·kb·(5 − tr H)
(Π — імпульс, Φ̇ = (N/a²)Π; E = U̇Uᵀ — швидкість, інерція ланки ka/N_l). s заморожене (пробне поле).
Передбачення континууму: c = N/a, сповільнення часу ω ∝ N, статична енергія солітона = N·m, прискорення речовини −∇N, світло n = a/N."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import gauge91 as G
from gauge91 import sh, put, T_, PA, tr, DIRS
from phi90 import sym0, dV

class GaugeM(G.Gauge):
    def __init__(s, Nn, q=1.0, ka=1.0, kb=2.0, a=-1.0):
        L = Nn.shape[0]
        an = Nn**(-q)
        s.mP = an**2/Nn                                            # інерція Φ★ у вузлі
        s.cV = Nn*an**2                                            # множник потенціалу
        Nl = np.array([0.5*(Nn + sh(Nn, d)) for d in DIRS])
        Nt = np.array([(Nn + sh(Nn, DIRS[0]) + sh(Nn, DIRS[1]))/3, (sh(Nn, DIRS[0]) + sh(Nn, (1, 1)) + sh(Nn, DIRS[1]))/3])
        super().__init__(np.full((L, L), a), 1.0/Nl, Nt**(1 + 2*q), ka, kb)   # wl = 1/N_l (інерція ланки), wt = N_t/a_t²
        s.fl = Nl; s.Nn = Nn
    def forces(s, P, U):
        FP = -s.cV[..., None, None]*dV(P, s.a, s.b)
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
        return sym0(FP), FE
    def edens(s, P, Pi, U, E):
        e = 0.5*np.sum(Pi**2, axis=(-2, -1))/s.mP
        P2 = P @ P; t2 = tr(P2)
        e += s.cV*(s.a/2*t2 - s.b/3*tr(P2 @ P) + t2**2/4 + 0.5/4*tr(P2 @ P2))
        for m, d in enumerate(DIRS):
            D = U[m] @ sh(P, d) @ T_(U[m]) - P
            e += s.fl[m]*(1/3)*np.sum(D**2, axis=(-2, -1)) + s.ka*s.wl[m]/4*np.sum(E[m]**2, axis=(-2, -1))
        Hu, Hd = s.hol(U)
        e += s.kb*(s.wt[0]*(5 - tr(Hu)) + s.wt[1]*(5 - tr(Hd)))
        return e
    def step(s, P, Pi, U, E, dt, gam=0.0):
        iv = (1.0/s.mP)[..., None, None]
        FP, FE = s.forces(P, U)
        Pi = Pi + 0.5*dt*(FP - gam*Pi); E = E + 0.5*dt*(FE - gam*E)
        P = sym0(P + dt*iv*Pi); U = G.cayley(dt*E) @ U
        FP, FE = s.forces(P, U)
        Pi = (Pi + 0.5*dt*FP)/(1 + 0.5*dt*gam); E = (E + 0.5*dt*FE)/(1 + 0.5*dt*gam)
        return P, Pi, U, E
