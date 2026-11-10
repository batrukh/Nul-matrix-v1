"""(321) директор для живої моделі: GD (312) + GDW (319, член ланок зважено w_l)."""

import numpy as np, sys
import gauge91 as G
from gauge91 import sh, put, T_, PA, DIRS, cayley
from phi90 import sym0
class GD(G.Gauge):
    def __init__(s, a, wl, wt, ka=1.0, kb=1.0, b=0.0, mu=1.0, k=0.09):
        super().__init__(a, wl, wt, ka, kb, b); s.mu, s.k = mu, k
    def edens_n(s, P, n, pn=None):
        e = s.mu*np.einsum('...i,...ij,...j', n, P @ P, n)
        if pn is not None: e = e + 0.5*np.sum(pn*pn, -1)
        for m, d in enumerate(DIRS):
            f = np.einsum('...i,...ij,...j', n, U_[m], sh(n, d)) if False else None
        return e
    def en(s, P, U, n):
        e = s.mu*np.einsum('...i,...ij,...j', n, P @ P, n)
        for m, d in enumerate(DIRS):
            f = np.einsum('...i,...i', n, np.einsum('...ij,...j', U[m], sh(n, d))); e = e - s.k/2*f**2
        return e
    def forces_d(s, P, U, n):
        FP, FE = s.forces(P, U)
        Pn = np.einsum('...ij,...j', P, n)
        Fn = -2*s.mu*np.einsum('...ij,...j', P, Pn)
        FP = FP - s.mu*sym0(np.einsum('...i,...j', Pn, n) + np.einsum('...i,...j', n, Pn))
        for m, d in enumerate(DIRS):
            X = np.einsum('...ij,...j', U[m], sh(n, d)); f = np.sum(n*X, -1)[..., None]
            Fn += s.k*f*X
            Fn += put(s.k*f*np.einsum('...ji,...j', U[m], n), d)
            R = -s.k*f[..., None]*np.einsum('...i,...j', n, X)
            FE[m] += -(2/(s.ka*s.wl[m]))[..., None, None]*PA(R)
        return FP, FE, Fn
    def step_d(s, P, Pi, U, E, n, pn, dt, gam=0.0):
        FP, FE, Fn = s.forces_d(P, U, n)
        tang = lambda v, n: v - np.sum(v*n, -1, keepdims=True)*n
        Pi = Pi + 0.5*dt*(FP - gam*Pi); E = E + 0.5*dt*(FE - gam*E); pn = pn + 0.5*dt*(tang(Fn, n) - gam*pn)
        P = sym0(P + dt*Pi); U = cayley(dt*E) @ U; n = n + dt*pn; n /= np.linalg.norm(n, axis=-1, keepdims=True); pn = tang(pn, n)
        FP, FE, Fn = s.forces_d(P, U, n)
        f = 1 + 0.5*dt*gam
        Pi = (Pi + 0.5*dt*FP)/f; E = (E + 0.5*dt*FE)/f; pn = (pn + 0.5*dt*tang(Fn, n))/f
        return P, Pi, U, E, n, pn

class GDW(GD):
    def en(s, P, U, n):
        e = s.mu*np.einsum('...i,...ij,...j', n, P @ P, n)
        for m, d in enumerate(DIRS):
            f = np.einsum('...i,...i', n, np.einsum('...ij,...j', U[m], sh(n, d))); e = e - s.wl[m]*s.k/2*f**2
        return e
    def forces_d(s, P, U, n):
        FP, FE = s.forces(P, U); Pn = np.einsum('...ij,...j', P, n); Fn = -2*s.mu*np.einsum('...ij,...j', P, Pn)
        FP = FP - s.mu*sym0(np.einsum('...i,...j', Pn, n) + np.einsum('...i,...j', n, Pn))
        for m, d in enumerate(DIRS):
            kk = (s.k*s.wl[m])[..., None]
            X = np.einsum('...ij,...j', U[m], sh(n, d)); f = np.sum(n*X, -1)[..., None]
            Fn += kk*f*X; Fn += put(kk*f*np.einsum('...ji,...j', U[m], n), d)
            R_ = -(kk*f)[..., None]*np.einsum('...i,...j', n, X)
            FE[m] += -(2/(s.ka*s.wl[m]))[..., None, None]*PA(R_)
        return FP, FE, Fn
