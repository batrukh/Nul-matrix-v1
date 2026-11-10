"""(298) комплексна нитка: калібрувальне SO(5)×U(1). Нитка = поворот U ∈ SO(5) і фаза e^{i q a}; Φ★ комплексне симетричне безслідове.
    H = Σ_n [½|Π|² + V(Φ)] + Σ_l [(1/3)|e^{iqa}UΦ'Uᵀ − Φ|² + (ka w/4)|E|² + (kc w/2) ȧ²] + Σ_tri [kb w (5 − tr H) + kp w (1 − cos F)]
    V = a/2 t + c/4 t² + d/4 tr M² + μ/4 (t² − |tr Φ²|²), t = tr Φ†Φ, M = Φ†Φ  (b = 0: кубічний член несумісний з фазою).
При a ≡ 0 і дійсному Φ — точно gauge91 з b = 0."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
from gauge91 import DIRS, sh, put, cayley, weights, gen, T_, PA, tr
I5 = np.eye(5)
def sym0(P):
    P = 0.5*(P + T_(P)); return P - tr(P)[..., None, None]*I5/5
H_ = lambda A: np.conj(T_(A))

class CGauge:
    def __init__(s, a, wl, wt, ka=1., kb=2., kc=1., kp=2., q=1., mu=0.3, c=1., d=0.5):
        s.a, s.wl, s.wt, s.ka, s.kb, s.kc, s.kp, s.q, s.mu, s.c, s.d = a, wl, wt, ka, kb, kc, kp, q, mu, c, d
    def hol(s, U):
        U0, U1, U2 = U
        return U0 @ sh(U2, DIRS[0]) @ T_(U1), sh(U1, DIRS[0]) @ T_(sh(U0, DIRS[1])) @ T_(sh(U2, DIRS[0]))
    def flux(s, A):
        A0, A1, A2 = A
        return A0 + sh(A2, DIRS[0]) - A1, sh(A1, DIRS[0]) - sh(A0, DIRS[1]) - sh(A2, DIRS[0])
    def edens(s, P, Pi, U, E, A, EA):
        e = 0.5*np.sum(np.abs(Pi)**2, axis=(-2, -1))
        M = H_(P) @ P; t = tr(M).real; s2 = tr(P @ P)
        e += s.a/2*t + s.c/4*t**2 + s.d/4*tr(M @ M).real + s.mu/4*(t**2 - np.abs(s2)**2)
        for m, dd in enumerate(DIRS):
            D = np.exp(1j*s.q*A[m])[..., None, None]*(U[m] @ sh(P, dd) @ T_(U[m])) - P
            e += (1/3)*np.sum(np.abs(D)**2, axis=(-2, -1)) + s.ka*s.wl[m]/4*np.sum(E[m]**2, axis=(-2, -1)) + s.kc*s.wl[m]/2*EA[m]**2
        Hu, Hd = s.hol(U); Fu, Fd = s.flux(A)
        e += s.kb*(s.wt[0]*(5 - tr(Hu)) + s.wt[1]*(5 - tr(Hd))) + s.kp*(s.wt[0]*(1 - np.cos(Fu)) + s.wt[1]*(1 - np.cos(Fd)))
        return e
    def forces(s, P, U, A):
        M = H_(P) @ P; t = tr(M).real[..., None, None]; s2 = tr(P @ P)[..., None, None]
        FP = -(s.a[..., None, None]*P + s.c*t*P + s.d*(P @ M) + s.mu*(t*P - s2*np.conj(P)))
        R = [np.zeros_like(U[0]) for _ in range(3)]; FA = [np.zeros_like(A[0]) for _ in range(3)]
        for m, dd in enumerate(DIRS):
            cph = np.exp(1j*s.q*A[m])[..., None, None]
            X = U[m] @ sh(P, dd) @ T_(U[m]); cX = cph*X; D = cX - P
            FP += (2/3)*D
            FP -= put((2/3)*np.conj(cph)*(T_(U[m]) @ D @ U[m]), dd)
            Dc = np.conj(D)
            R[m] += (2/3)*np.real(Dc @ cX - cX @ Dc)
            FA[m] += (2*s.q/3)*np.imag(np.sum(Dc*cX, axis=(-2, -1)))
        U0, U1, U2 = U
        bu = (s.kb*s.wt[0])[..., None, None]; bd = (s.kb*s.wt[1])[..., None, None]
        Aa, B = U0, sh(U2, DIRS[0]); Hu = Aa @ B @ T_(U1)
        R[0] += -bu*T_(Hu); R[2] += put(-bu*(T_(Aa) @ T_(Hu) @ Aa), DIRS[0]); R[1] += -bu*Hu
        Aa, B, C = sh(U1, DIRS[0]), sh(U0, DIRS[1]), sh(U2, DIRS[0]); Hd = Aa @ T_(B) @ T_(C)
        R[1] += put(-bd*T_(Hd), DIRS[0]); R[0] += put(-bd*(T_(C) @ Hd @ C), DIRS[1]); R[2] += put(-bd*Hd, DIRS[0])
        Fu, Fd = s.flux(A); su = s.kp*s.wt[0]*np.sin(Fu); sd = s.kp*s.wt[1]*np.sin(Fd)
        FA[0] -= su; FA[2] -= put(su, DIRS[0]); FA[1] += su
        FA[1] -= put(sd, DIRS[0]); FA[0] += put(sd, DIRS[1]); FA[2] += put(sd, DIRS[0])
        FE = np.array([-(2/(s.ka*s.wl[m]))[..., None, None]*PA(R[m]) for m in range(3)])
        FEA = np.array([FA[m]/(s.kc*s.wl[m]) for m in range(3)])
        return sym0(FP), FE, FEA
    def step(s, P, Pi, U, E, A, EA, dt, gam=0.0):
        FP, FE, FEA = s.forces(P, U, A)
        Pi = Pi + 0.5*dt*(FP - gam*Pi); E = E + 0.5*dt*(FE - gam*E); EA = EA + 0.5*dt*(FEA - gam*EA)
        P = sym0(P + dt*Pi); U = cayley(dt*E) @ U; A = A + dt*EA
        FP, FE, FEA = s.forces(P, U, A)
        g = 1 + 0.5*dt*gam
        return P, (Pi + 0.5*dt*FP)/g, U, (E + 0.5*dt*FE)/g, A, (EA + 0.5*dt*FEA)/g
