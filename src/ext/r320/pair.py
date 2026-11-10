"""(320) Пара вихорів Аліси в повній моделі (Φ★ + калібрувальне SO(5) + директор n, база κa 1, κb 2, b 0,1; μ 0,03, k 0,09).
Анзац як (305)/(313): Φ = R(θ/2)P0R(θ/2)ᵀ, n = R(θ/2)e2, ланки rot23(−(1−f)Δθ/2), f = exp(−(r/r0)²); R — поворот у площині (e2, e3),
θ = arg(x − v1) − arg(x − v2), v1,2 = c ± R(cos φ, sin φ). Закріплення: ями a(x) = −1 + A·Σ exp(−|x − v_i|²/w²).
Обмін φ 0 → π: конфігурація в π = G·(конфігурація в 0)·G, G = diag(1,1,1,−1,−1) — точно (той самий набір ям)."""
import numpy as np, sys
sys.path.insert(0, '/mnt/project-files/model/s52/r320'); from gd import GD
sys.path.insert(0, '/mnt/project-files/model/s52/r268/code'); import phi90 as PH
DIRS = [(1, 0), (0, 1), (-1, 1)]
MU, KD, KA, KB, B = 0.03, 0.09, 1.0, 2.0, 0.1
zv = np.load('/mnt/project-files/model/s52/r312/fin_vac0.03.npz'); _w, _v = np.linalg.eigh(zv['P'][0, 0])
P0 = np.diag(_w); NV = _v.T @ zv['n'][0, 0]; assert abs(abs(NV[2]) - 1) < 1e-6, NV        # n уздовж e2 (−0,264)
E2 = np.eye(5)[2]
def rot23(a):
    M = np.broadcast_to(np.eye(5), a.shape + (5, 5)).copy(); c, s = np.cos(a), np.sin(a)
    M[..., 2, 2] = c; M[..., 3, 3] = c; M[..., 2, 3] = -s; M[..., 3, 2] = s; return M
def pos(L):
    x, y = np.meshgrid(np.arange(L), np.arange(L), indexing='ij'); return np.stack([x + 0.5*y, np.sqrt(3)/2*y], -1)
def wrapv(D, L):                                     # найближчий образ на косокутному торі
    best = None
    for p in (-1, 0, 1):
        for q in (-1, 0, 1):
            E = D + np.array([p*L + q*L/2, q*np.sqrt(3)/2*L]); r = np.hypot(E[..., 0], E[..., 1])
            if best is None: best = (r, E)
            else: k = r < best[0]; best = (np.where(k, r, best[0]), np.where(k[..., None], E, best[1]))
    return best
def centre(L): return np.array([0.75*L + 0.25, np.sqrt(3)/4*L + 0.2])
def verts(L, R, phi):
    c = centre(L); e = np.array([np.cos(phi), np.sin(phi)]); return c + R*e, c - R*e
def ansatz(L, R, phi, r0=3.0):
    S = pos(L); v1, v2 = verts(L, R, phi)
    def th(Q):
        a1 = wrapv(Q - v1, L)[1]; a2 = wrapv(Q - v2, L)[1]
        return np.arctan2(a1[..., 1], a1[..., 0]) - np.arctan2(a2[..., 1], a2[..., 0])
    t = th(S); Rn = rot23(t/2); P = Rn @ P0 @ np.swapaxes(Rn, -1, -2); n = Rn @ E2
    U = np.empty((3, L, L, 5, 5))
    for m, d in enumerate(DIRS):
        S2 = S + np.array([d[0] + 0.5*d[1], np.sqrt(3)/2*d[1]]); M = 0.5*(S + S2)
        dth = np.angle(np.exp(1j*(th(S2) - t)))
        r = np.minimum(wrapv(M - v1, L)[0], wrapv(M - v2, L)[0]); f = np.exp(-(r/r0)**2)
        U[m] = rot23(-(1 - f)*dth/2)
    return P, U, n
def cross(p1, p2, q1, q2):
    d = lambda a, b, c: (b[..., 0] - a[..., 0])*(c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1])*(c[..., 0] - a[..., 0])
    return (d(p1, p2, q1)*d(p1, p2, q2) < 0) & (d(q1, q2, p1)*d(q1, q2, p2) < 0)
Z5 = np.diag([1, 1, -1, -1, 1.0])
def ansatz2(L, R, phi, r0=3.0):
    """(320, виправлено) калібрування з тонким розрізом (як thick у (305)): Φ = P0, n = e2 скрізь; ланки = Z на розрізі v1–v2,
    × rot23(f·Δθ/2) біля ядер (f = exp(−(r/r0)²)) — без шва на торі, далеке поле — чистий вакуум."""
    S = pos(L); v1, v2 = verts(L, R, phi)
    def th(Q):
        a1 = wrapv(Q - v1, L)[1]; a2 = wrapv(Q - v2, L)[1]
        return np.arctan2(a1[..., 1], a1[..., 0]) - np.arctan2(a2[..., 1], a2[..., 0])
    t = th(S); U = np.empty((3, L, L, 5, 5))
    for m, d in enumerate(DIRS):
        S2 = S + np.array([d[0] + 0.5*d[1], np.sqrt(3)/2*d[1]]); M = 0.5*(S + S2)
        dth = np.angle(np.exp(1j*(th(S2) - t)))
        r = np.minimum(wrapv(M - v1, L)[0], wrapv(M - v2, L)[0]); f = np.exp(-(r/r0)**2)
        U[m] = rot23(f*dth/2)
        c = cross(v1, v2, S, S2); U[m][c] = Z5 @ U[m][c]
    return np.broadcast_to(P0, (L, L, 5, 5)).copy(), U, np.broadcast_to(E2, (L, L, 5)).copy()
def model(L, R, phi, A=0.5, w=2.0):
    S = pos(L); v1, v2 = verts(L, R, phi)
    a = -1.0 + A*(np.exp(-(wrapv(S - v1, L)[0]/w)**2) + np.exp(-(wrapv(S - v2, L)[0]/w)**2))
    return GD(a, np.ones((3, L, L)), np.ones((2, L, L)), KA, KB, B, mu=MU, k=KD)
G5 = np.diag([1, 1, 1, -1, -1.0])
