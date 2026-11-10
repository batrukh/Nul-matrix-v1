"""(298) вакуум комплексного Φ★ (симетричне безслідове, комплексне): V = a/2 t + c/4 t² + d/4 tr(M²) + μ/4 (t² − |tr P²|²), t = tr P†P, M = P†P."""
import numpy as np, sys
I5 = np.eye(5)
def sym0(P):
    P = 0.5*(P + P.swapaxes(-1, -2)); return P - np.trace(P, axis1=-2, axis2=-1)[..., None, None]*I5/5
def V(P, a=-1., c=1., d=.5, mu=0.):
    M = P.conj().T @ P; t = np.trace(M).real; s = np.trace(P @ P)
    return a/2*t + c/4*t**2 + d/4*np.trace(M @ M).real + mu/4*(t**2 - abs(s)**2)
def G(P, a=-1., c=1., d=.5, mu=0.):
    M = P.conj().T @ P; t = np.trace(M).real; s = np.trace(P @ P)
    return sym0(a*P + c*t*P + d*P @ M + mu*(t*P - s*P.conj()))
def relax(mu, seed, n=40000, lr=0.02):
    r = np.random.default_rng(seed); P = sym0(r.standard_normal((5, 5)) + 1j*r.standard_normal((5, 5)))*0.3
    for _ in range(n): P = P - lr*G(P, mu=mu)
    return P
if __name__ == '__main__':
    for mu in [0., 0.1, 0.3, 1.0]:
        res = []
        for s in range(12):
            P = relax(mu, s); e = V(P, mu=mu)
            sv = np.linalg.svd(P, compute_uv=False); s2 = np.trace(P @ P); t = np.trace(P.conj().T @ P).real
            res.append((e, np.round(np.sort(sv), 4), abs(s2)/t))
        res.sort(key=lambda x: x[0])
        print(f'mu={mu}: best E={res[0][0]:.6f} sv={res[0][1]} |trP2|/t={res[0][2]:.4f}; worst E={res[-1][0]:.6f} sv={res[-1][1]}')
    # дійсний 3+2 з b=0
    P = np.diag([-1, -1, -1, 1.5, 1.5]); 
    for _ in range(40000): P = P - 0.02*G(P.astype(complex)).real
    print('real 3+2:', np.round(np.diag(P), 4), V(P.astype(complex)))
