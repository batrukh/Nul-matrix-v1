"""(314) № 4 «динамічне сусідство», пробний крок: ціна перемикання ребра (Пахнер 2→2) у полі бази.
Ромб a,b,c,d: старе ребро b–c (трикутники abc, bcd) → нове a–d (abd, acd). Нова ланка T(a←d) = H^{1/2}·P2,
H = P1·P2ᵀ — голономія ромба (P1 = T(a←b)T(b←d), P2 = T(a←c)T(c←d)): обидва нові трикутники мають H^{1/2}.
ΔE = (1/3)(|T Φ_d Tᵀ − Φ_a|² − |T Φ_c Tᵀ − Φ_b|²) + κb[2(5 − tr H^{1/2}) − (5 − tr H1) − (5 − tr H2)].
У вакуумі (Φ стале, U = 1) ΔE = 0 — сітку нічого не тримає. python3 flip.py NPZ OUT"""
import numpy as np, sys, json
from scipy.linalg import schur
sys.path.insert(0, '/mnt/project-files/model/s52/r268/code'); import phi90 as PH
z = np.load(sys.argv[1]); P, U = z['P'], z['U']; L = P.shape[0]; KB = 2.0
D = [(1, 0), (0, 1), (-1, 1)]
def T(x, y):                                  # перенесення з y у x (сусіди)
    for m, d in enumerate(D):
        if ((x[0] + d[0]) % L, (x[1] + d[1]) % L) == y: return U[m][x]
        if ((y[0] + d[0]) % L, (y[1] + d[1]) % L) == x: return U[m][y].T
    raise ValueError
def sqrt_orth(H):
    Tm, Z = schur(H, output='real'); S = np.zeros_like(Tm); i = 0
    while i < 5:
        if i < 4 and abs(Tm[i + 1, i]) > 1e-12:
            a = np.arctan2(Tm[i + 1, i], Tm[i, i])/2; S[i:i+2, i:i+2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]; i += 2
        else:
            S[i, i] = 1.0 if Tm[i, i] > 0 else 1.0; 
            if Tm[i, i] < 0: S[i, i] = np.nan
            i += 1
    return Z @ S @ Z.T
ad = lambda p, q: ((p[0] + q[0]) % L, (p[1] + q[1]) % L)
X, Y = PH.coords(L); v = [(float(z['cx']), float(z['cy'])), (float(z['cx']), float(z['vy2']))]
out = np.full((3, L, L), np.nan); rr = np.zeros((3, L, L)); nflip = 0
for m, d in enumerate(D):
    # ребро b=(i,j) → c=b+d; треті вершини двох трикутників
    third = {0: [(0, 1), (1, -1)], 1: [(-1, 1), (1, 0)], 2: [(0, 1), (-1, 0)]}[m]
    for i in range(L):
        for j in range(L):
            b = (i, j); c = ad(b, d); a = ad(b, third[0]); dd = ad(b, third[1])
            P1 = T(a, b) @ T(b, dd); P2 = T(a, c) @ T(c, dd); H = P1 @ P2.T
            H1 = T(a, b) @ T(b, c) @ T(c, a); H2 = T(dd, b) @ T(b, c) @ T(c, dd)
            S = sqrt_orth(H)
            if np.isnan(S).any(): nflip += 1; continue
            Tn = S @ P2
            e_new = np.sum((Tn @ P[dd] @ Tn.T - P[a])**2)/3; Tb = T(b, c); e_old = np.sum((Tb @ P[c] @ Tb.T - P[b])**2)/3
            out[m, i, j] = e_new - e_old + KB*(2*(5 - np.trace(S)) - (5 - np.trace(H1)) - (5 - np.trace(H2)))
            mx = 0.25*(X[a] + X[dd] + X[b] + X[c]); my = 0.25*(Y[a] + Y[dd] + Y[b] + Y[c])   # приблизно (без обгортки)
            rr[m, i, j] = min(PH.tdist(np.array(X[b]), np.array(Y[b]), vx, vy, L) for vx, vy in v)
np.savez(sys.argv[2], dE=out, r=rr); fin = out[np.isfinite(out)]
print('ребер', fin.size, 'неоднозначних √', nflip, 'ΔE min/max', fin.min(), fin.max())
for lo, hi in [(0, 2), (2, 4), (4, 6), (6, 10), (10, 20), (20, 99)]:
    s = (rr >= lo) & (rr < hi) & np.isfinite(out); print(f'r {lo}-{hi}: n {s.sum()}, ΔE min {out[s].min():.4g}, середнє {out[s].mean():.4g}, частка ΔE<0 {np.mean(out[s] < -1e-9):.3f}')
