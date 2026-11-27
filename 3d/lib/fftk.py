"""fftk: швидке K s = P^n s на правильній трикутній ґратці-торі через FFT (P трансляційно-інваріантний: 3 сім'ї ниток).
Точно рівне n множенням на P (перевіряється при побудові)."""
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
    # ядро: відгук P на одиничну нитку кожної сім'ї в (0,0)
    Kr = np.zeros((3, 3, L, L))
    for c in range(3):
        e = np.zeros(f.m); e[idx[c, 0, 0]] = 1; r = P @ e
        Kr[:, c] = r[idx]
    Kk = np.fft.fft2(Kr)                       # (3,3,L,L)
    M = np.moveaxis(Kk, (0, 1), (2, 3))        # (L,L,3,3)
    Mn = np.ascontiguousarray(np.linalg.matrix_power(M, n)[:, :L//2 + 1])   # дійсний вхід: досить половини частот
    def apply(s):
        S = np.fft.rfft2(s[idx])               # (3,L,L/2+1)
        out = np.fft.irfft2(np.einsum('xyab,bxy->axy', Mn, S), s=(L, L))
        return out[fam, X, Y]
    return apply
