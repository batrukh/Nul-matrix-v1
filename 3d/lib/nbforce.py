"""nbforce: сила Sys.F (stall43 + M34.rhs з sim34d) одним компільованим проходом (numba). Та сама формула, той самий порядок
підсумовування по сім'ях трикутників; збіг зі звичайним кодом — до похибки округлення (перевіряється в test_nb)."""
import numpy as np
from numba import njit

@njit(cache=True)
def _force(zr, zi, psi, tail, head, E1, S1, E2, S2, E3, S3, TV, g, mu, beta, eps0, h, kappa, d2, wp, lc, m0, csr, csi, ccr, cci):
    # cci — масив по нитках (82): поворот 1 + iα_e може бути різним для різних ниток (плавна стріла)
    m = zr.shape[0]; n = psi.shape[0]; nt = TV.shape[0]
    z = zr + 1j*zi
    A2 = zr*zr + zi*zi
    Ad = np.sqrt(A2 + d2)
    chi = ((1.0 + psi)/2.0)**2; dchi = (1.0 + psi)/2.0
    lap = np.zeros(n); force = np.empty(n)
    dps = np.empty(m)
    for e in range(m):
        dp = psi[head[e]] - psi[tail[e]]; dps[e] = dp
        we = Ad[e]/wp
        lap[tail[e]] += we*dp
    lap2 = np.zeros(n)
    for e in range(m):
        lap2[head[e]] += Ad[e]/wp*dps[e]
    for v in range(n):
        force[v] = kappa*(lap[v] - lap2[v]) - (psi[v]**3 - psi[v]) + h
    # трикутники
    b1r = np.zeros(m); b1i = np.zeros(m); b2r = np.zeros(m); b2i = np.zeros(m); b3r = np.zeros(m); b3i = np.zeros(m)
    w1r = np.zeros(m); w1i = np.zeros(m); w2r = np.zeros(m); w2i = np.zeros(m); w3r = np.zeros(m); w3i = np.zeros(m)
    cost = np.empty(nt)
    for t in range(nt):
        p = TV[t, 0]; q = TV[t, 1]; r = TV[t, 2]
        Et = eps0*(chi[p] + chi[q] + chi[r])/3.0
        l1 = z[E1[t]] if S1[t] > 0 else np.conj(z[E1[t]])
        l2 = z[E2[t]] if S2[t] > 0 else np.conj(z[E2[t]])
        l3 = z[E3[t]] if S3[t] > 0 else np.conj(z[E3[t]])
        P = l1*l2*l3; aP = np.sqrt(P.real*P.real + P.imag*P.imag + d2); u1 = P/aP - 1.0
        c = np.conj(l2*l3)*u1; v = -Et*(c if S1[t] > 0 else np.conj(c)); b1r[E1[t]] += v.real; b1i[E1[t]] += v.imag
        c = np.conj(l3*l1)*u1; v = -Et*(c if S2[t] > 0 else np.conj(c)); b2r[E2[t]] += v.real; b2i[E2[t]] += v.imag
        c = np.conj(l1*l2)*u1; v = -Et*(c if S3[t] > 0 else np.conj(c)); b3r[E3[t]] += v.real; b3i[E3[t]] += v.imag
        c = np.conj(l3*l2); v = c if S1[t] > 0 else np.conj(c); w1r[E1[t]] += v.real; w1i[E1[t]] += v.imag
        c = np.conj(l1*l3); v = c if S2[t] > 0 else np.conj(c); w2r[E2[t]] += v.real; w2i[E2[t]] += v.imag
        c = np.conj(l2*l1); v = c if S3[t] > 0 else np.conj(c); w3r[E3[t]] += v.real; w3i[E3[t]] += v.imag
        cost[t] = (aP - P.real)/3.0*eps0
    for k in range(3):
        for t in range(nt):
            nd = TV[t, k]; force[nd] -= dchi[nd]*cost[t]
    fc = np.zeros(n)
    dz = np.empty(m, np.complex128)
    cs = csr + 1j*csi
    for e in range(m):
        brk = (0.0 + 0.0j) + (b1r[e] + 1j*b1i[e]) + (b2r[e] + 1j*b2i[e]) + (b3r[e] + 1j*b3i[e])
        ww = (0.0 + 0.0j) + (w1r[e] + 1j*w1i[e]) + (w2r[e] + 1j*w2i[e]) + (w3r[e] + 1j*w3i[e])
        ph = z[e]/Ad[e]
        dzf = -g*z[e] - mu*A2[e]*z[e] + beta*ww + brk - 0.5*kappa*dps[e]**2/wp*ph
        sf = -(g + mu*A2[e])*z[e]; cpart = dzf - sf
        pe = (psi[tail[e]] + psi[head[e]])/2.0
        sf = sf + lc*pe*z[e]/(2.0*wp**2)
        dz[e] = cs*sf + (ccr + 1j*cci[e])*cpart
        qv = lc*(A2[e] - m0)/(4.0*wp**2)
        fc[tail[e]] += qv
    fc2 = np.zeros(n)
    for e in range(m):
        fc2[head[e]] += lc*(A2[e] - m0)/(4.0*wp**2)
    dpsi = force + (fc + fc2)
    return dz, dpsi

def force(s, z, psi):
    f, mod = s.fab, s.mod
    assert not mod.pin.any()
    return _force(np.ascontiguousarray(z.real), np.ascontiguousarray(z.imag), psi, f.tail, f.head, f.E1, f.S1, f.E2, f.S2, f.E3, f.S3,
                  f.TV, 1.0, 1.0, mod.beta, mod.eps0, mod.h, mod.kappa, mod.d2, s.wp, s.lc, s.m0, s.cs.real, s.cs.imag, s.cc.real,
                  getattr(s, 'acv', None) if getattr(s, 'acv', None) is not None else np.full(f.m, s.cc.imag))
