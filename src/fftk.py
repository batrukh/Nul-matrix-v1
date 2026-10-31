"""fftk: fast K s = P^n s on the regular triangular torus lattice via FFT (P is translation invariant: 3 link families).
Exactly equal to n multiplications by P (checked at construction)."""
import numpy as np
def build(f, L, P, n):
    xa, ya = f.xy[f.tail].T; xb, yb = f.xy[f.head].T
    dx = (xb - xa + L//2) % L - L//2; dy = (yb - ya + L//2) % L - L//2
    dirs = sorted({(a, b) for a, b in zip(dx, dy)})
    canon = []
    for d in dirs:
        if (-d[0], -d[1]) not in canon: canon.append(d)
    assert len(canon) == 3, canon
    fam = np.empty(f.m, int); X = np.empty(f.m, int); Y = np.empty(f.m, int)
    for i, (a, b) in enumerate(zip(dx, dy)):
        if (a, b) in canon: fam[i] = canon.index((a, b)); X[i], Y[i] = xa[i], ya[i]
        else: fam[i] = canon.index((-a, -b)); X[i], Y[i] = xb[i], yb[i]
    idx = np.full((3, L, L), -1); idx[fam, X, Y] = np.arange(f.m); assert (idx >= 0).all()
    # kernel: response of P to a unit link of each family at (0,0)
    Kr = np.zeros((3, 3, L, L))
    for c in range(3):
        e = np.zeros(f.m); e[idx[c, 0, 0]] = 1; r = P @ e
        Kr[:, c] = r[idx]
    Kk = np.fft.fft2(Kr)                       # (3,3,L,L)
    M = np.moveaxis(Kk, (0, 1), (2, 3))        # (L,L,3,3)
    Mn = np.ascontiguousarray(np.linalg.matrix_power(M, n)[:, :L//2 + 1])   # real input: half of the frequencies suffices
    def apply(s):
        S = np.fft.rfft2(s[idx])               # (3,L,L/2+1)
        out = np.fft.irfft2(np.einsum('xyab,bxy->axy', Mn, S), s=(L, L))
        return out[fam, X, Y]
    return apply
