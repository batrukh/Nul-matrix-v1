"""pfcB (22.11, (458) етап Б): PFC (421) зі змінним eps(x) — затвор росту. Лінійна частина з eps0=0,25 неявно, (eps(x)-eps0)psi — явно.
Кристалічність c(x) = згладжене (psi-psibar)^2 / U_C (U_C=0,0694 — повна ОЦК (448), eps0,25, psibar -0,30)."""
import sys, numpy as np
R0="/mnt/project-files/model/s52"; sys.path[:0]=[R0+"/r421"]
from pfc import kgrid, A_BCC
CELLS, NG, PB, EPS0, U_C = 13, 104, -0.30, 0.25, 0.0694
LB = CELLS*A_BCC; DX = LB/NG
K2 = kgrid((NG,)*3, (LB,)*3)
def gauss_k(sig): return np.exp(-0.5*K2*sig*sig)
GC = gauss_k(0.5*A_BCC)            # згладжування кристалічності: півкомірки
def step(psi, epsx, dt, nst, C=1.0):
    den = 1 + dt*K2*(-EPS0 + (1 - K2)**2 + C); ph = np.fft.fftn(psi); de = epsx - EPS0
    for _ in range(nst):
        p = np.real(np.fft.ifftn(ph)); ph = (ph - dt*K2*np.fft.fftn(p**3 - C*p - de*p))/den
    return np.real(np.fft.ifftn(ph))
def cryst(psi): return np.real(np.fft.ifftn(np.fft.fftn((psi - PB)**2)*GC))/U_C
def seedball(rng, cen, rad, noise=0.02):
    x = np.arange(NG)*DX; X, Y, Z = np.meshgrid(x, x, x, indexing='ij'); q = 2*np.pi/A_BCC
    d = [((V - c + LB/2) % LB) - LB/2 for V, c in zip((X, Y, Z), cen)]; r = np.sqrt(d[0]**2 + d[1]**2 + d[2]**2)
    m = 0.5*(1 - np.tanh((r - rad)/2.0))
    V6 = np.array([(1,1,0),(1,-1,0),(0,1,1),(0,1,-1),(1,0,1),(1,0,-1)], float)
    p = 0.5*sum(np.cos(q*(v[0]*d[0] + v[1]*d[1] + v[2]*d[2])) for v in V6)
    psi = PB + noise*rng.standard_normal((NG,)*3) + m*0.3*p/3.0; return psi - psi.mean() + PB
def splat(idx, sig):
    """індикатор зони біля вузлів idx (сітка), гаусове ядро з піком 1; повертає поле max~1."""
    g = np.zeros((NG,)*3); np.add.at(g, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
    return np.real(np.fft.ifftn(np.fft.fftn(g)*gauss_k(sig)))*(2*np.pi*sig*sig)**1.5/DX**3
