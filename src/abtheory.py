"""(285) Ааронов–Бом для плоскої хвилі e^{ikx} (зліва) на потоці α (Aharonov, Bohm 1959, повернуто на π): ψ = Σ_m (−1)^m (−i)^{|m+α|} J_{|m+α|}(kr) e^{imθ}."""
import numpy as np
from scipy.special import jv
def ab(x, y, k, alpha=0.5, M=200):
    r = np.hypot(x, y); th = np.arctan2(y, x); psi = np.zeros_like(r, dtype=complex)
    for m in range(-M, M + 1):
        nu = abs(m + alpha); psi += (-1)**m*np.exp(-1j*np.pi/2*nu)*jv(nu, k*r)*np.exp(1j*m*th)
    return psi
if __name__ == '__main__':
    for lam in (6, 8, 12):
        k = 2*np.pi/lam
        for d in (8, 16, 24):
            ys = np.linspace(-20, 20, 81); I = np.abs(ab(np.full_like(ys, d), ys, k))**2
            I0 = np.abs(ab(np.full_like(ys, d), ys, k, 0.0))**2
            w = np.abs(ys) < 3
            print(lam, d, 'I(y=0) %.3f' % I[40], 'середнє |y|<3: %.3f' % I[w].mean(), 'контроль %.3f' % I0[40], 'півширина провалу', ys[40 + np.argmax(I[40:] > 0.5)])

def ab_sim(x, y, k, alpha=0.5):
    """Те саме в калібруванні розрахунку: розріз — промінь +y від вихора (поле множиться на −1 при його перетині), A = 0 поза ним.
    ψ_sim = e^{+iαΘ}·ψ, Θ — кут з розрізом на +y, Θ ∈ (−3π/2, π/2]; нормовано на падаючу хвилю далеко ліворуч."""
    th = np.arctan2(y, x); Th = np.where(th > np.pi/2, th - 2*np.pi, th)
    ps = np.exp(1j*alpha*Th)*ab(x, y, k, alpha)
    ref = np.exp(1j*alpha*(-np.pi))*ab(np.array([-40.0]), np.array([0.0]), k, alpha)[0]/np.exp(1j*k*(-40.0))
    return ps/ref
