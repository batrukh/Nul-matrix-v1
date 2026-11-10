"""(305) Пара вихорів Аліси, посаджена калібрувально: Φ однорідне (вакуум бази), ланки, які перетинає відрізок v1–v2, = Z = diag(1,1,−1,−1,1).
Вихори — у центрах трикутників (вузли соти)."""
import numpy as np, sys
sys.path.insert(0, '/mnt/project-files/model/s52/r268/code'); import phi90 as PH
DIRS = [(1, 0), (0, 1), (-1, 1)]
Z = np.diag([1, 1, -1, -1, 1.0])
P0 = np.diag(np.sort(np.linalg.eigvalsh(PH.vacuum(-1.0, 0.1))))
def centers(L):
    x, y = np.meshgrid(np.arange(L), np.arange(L), indexing='ij'); X, Y = x + 0.5*y, np.sqrt(3)/2*y
    return np.stack([X + 0.5, Y + np.sqrt(3)/6], -1), np.stack([X, Y + np.sqrt(3)/3], -1)
def snap(L, p):
    A, B = centers(L); C = np.concatenate([A.reshape(-1, 2), B.reshape(-1, 2)])
    return C[np.argmin(((C - p)**2).sum(1))]
def cross(p1, p2, q1, q2):
    d = lambda a, b, c: (b[..., 0] - a[..., 0])*(c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1])*(c[..., 0] - a[..., 0])
    return (d(p1, p2, q1)*d(p1, p2, q2) < 0) & (d(q1, q2, p1)*d(q1, q2, p2) < 0)
def plant(L, v1, v2):
    x, y = np.meshgrid(np.arange(L), np.arange(L), indexing='ij'); S = np.stack([x + 0.5*y, np.sqrt(3)/2*y], -1)
    U = np.broadcast_to(np.eye(5), (3, L, L, 5, 5)).copy()
    for m, d in enumerate(DIRS):
        S2 = S + np.array([d[0] + 0.5*d[1], np.sqrt(3)/2*d[1]])
        c = cross(np.array(v1), np.array(v2), S, S2); U[m][c] = Z
    P = np.broadcast_to(P0, (L, L, 5, 5)).copy()
    return P, U

def rot23(a):
    M = np.broadcast_to(np.eye(5), a.shape + (5, 5)).copy(); c, s = np.cos(a), np.sin(a)
    M[..., 2, 2] = c; M[..., 3, 3] = c; M[..., 2, 3] = -s; M[..., 3, 2] = s; return M
def thick(L, v1, v2, r0):
    """рамкове калібрування: Φ = P0 скрізь, ланки = rot23(f·δ)·(Z на розрізі), δ = половина приросту кута пари,
    f = exp(−(r/r0)²): r0 → 0 — тонке ядро (лише Z), r0 → ∞ — неекранований (глобальний) вихор."""
    P, U = plant(L, v1, v2)
    x, y = np.meshgrid(np.arange(L), np.arange(L), indexing='ij'); S = np.stack([x + 0.5*y, np.sqrt(3)/2*y], -1)
    th = lambda Q: np.arctan2(Q[..., 1] - v1[1], Q[..., 0] - v1[0]) - np.arctan2(Q[..., 1] - v2[1], Q[..., 0] - v2[0])
    for m, d in enumerate(DIRS):
        S2 = S + np.array([d[0] + 0.5*d[1], np.sqrt(3)/2*d[1]]); M = 0.5*(S + S2)
        dth = np.angle(np.exp(1j*(th(S2) - th(S))))
        r = np.minimum(np.hypot(*(M - v1).T).T, np.hypot(*(M - v2).T).T)
        f = np.exp(-(r/r0)**2) if r0 > 0 else 0*r
        U[m] = U[m] @ rot23(f*dth/2)
    return P, U
