"""matter98: речовина у світі, крок 2 — чи живуть топологічні збудження протополя Φ★ (без калібрувального поля, однорідний світ a = −1).
Вакуум: власні значення u = −0,346 (×3), v = 0,518 (×2) → площина P (власний простір v) ∈ Gr(2,5).
Кандидати: (1) Z₂-вихор (π₁ = Z₂): поворот на θ/2 у площині (e3, e4); пара вихорів на торі.
           (2) згусток (π₂): площина = n^⊥ у R³ = span(e3, e4, e5), n — стереографічний образ (розмір λ), n → e3 на нескінченності.
Міряємо: Z₂-дефекти (трикутники з голономією det = −1), заряд Q (сума кутів голономії площини / 2π), розмір згустку, енергію.
Динаміка phi90: Φ̈ + γΦ̇ = ΔΦ − ∂V/∂Φ (Верле), спершу коротке тертя для очищення ядер."""
import numpy as np, json, sys
import phi90 as PH

W0 = np.linalg.eigvalsh(PH.vacuum(-1.0)); U_, V_ = W0[0], W0[-1]

def gen(i, j):
    G = np.zeros((5, 5)); G[i, j], G[j, i] = -1, 1; return G

def rot(phi, i, j):
    """exp(phi·G_ij) для поля кутів phi (L,L)"""
    c, s = np.cos(phi), np.sin(phi); R = np.broadcast_to(np.eye(5), phi.shape + (5, 5)).copy()
    R[..., i, i] = c; R[..., j, j] = c; R[..., j, i] = s; R[..., i, j] = -s; return R

def wrapd(X, Y, cx, cy, L):
    best = None
    for p in (-1, 0, 1):
        for q in (-1, 0, 1):
            dx, dy = X - cx + p*L + q*L/2, Y - cy + q*np.sqrt(3)/2*L; r = np.hypot(dx, dy)
            if best is None: best = (r, dx, dy)
            else:
                k = r < best[0]; best = (np.where(k, r, best[0]), np.where(k, dx, best[1]), np.where(k, dy, best[2]))
    return best

def init(kind, L, lam=4.0, d=16.0):
    X, Y = PH.coords(L); cx, cy = X.mean(), Y.mean(); P0 = np.diag([U_]*3 + [V_]*2)
    if kind == "vortex":
        dx1, dy1 = X - (cx - d/2), Y - cy; dx2, dy2 = X - (cx + d/2), Y - cy      # без згортання: розріз між вихорами, на краю тора лише мала дипольна нев'язка
        # кут, що на торі неперервний за модулем 2π для пари (+, −): θ1 − θ2
        th = np.arctan2(dy1, dx1) - np.arctan2(dy2, dx2)
        R = rot(th/2, 2, 3)
        return R @ P0 @ np.swapaxes(R, -1, -2)
    if kind == "lump":
        r, dx, dy = wrapd(X, Y, cx, cy, L); den = r**2 + lam**2
        n = np.stack([(r**2 - lam**2)/den, 2*lam*dx/den, 2*lam*dy/den], -1)      # компоненти на e3, e4, e5
        # м'яко повертаємо до вакууму на краю тора, щоб не було шва
        w = np.clip((L/2 - 4 - r)/4, 0, 1)[..., None]; n = n*w + np.array([1.0, 0, 0])*(1 - w); n /= np.linalg.norm(n, axis=-1, keepdims=True)
        P = np.zeros((L, L, 5, 5)); idx = [2, 3, 4]
        for a in range(3):
            for b in range(3):
                P[..., idx[a], idx[b]] = (a == b) - n[..., a]*n[..., b]
        return U_*np.eye(5) + (V_ - U_)*P
    raise ValueError(kind)

def planes(P):
    w, v = np.linalg.eigh(P); return v[..., :, 3:]          # 2 найбільші власні вектори — площина v

def polar2(M):
    u, s, vt = np.linalg.svd(M); return u @ vt

def holonomy(P):
    """для кожного трикутника: det голономії площини (−1 — Z₂-дефект) і кут (для заряду)"""
    B = planes(P); out_det, out_ang = [], []
    tri = [((0, 0), (1, 0), (0, 1)), ((1, 0), (1, 1), (0, 1))]
    sh = lambda A, d: np.roll(A, (-d[0], -d[1]), axis=(0, 1))
    for t in tri:
        b = [sh(B, d) for d in t]
        T = np.eye(2)
        for k in range(3):
            i, j = b[k], b[(k + 1) % 3]
            T = polar2(np.swapaxes(j, -1, -2) @ i) @ T
        out_det.append(np.linalg.det(T)); out_ang.append(np.arctan2(T[..., 1, 0], T[..., 0, 0]))
    return np.array(out_det), np.array(out_ang)

def lumpQ(P):
    """заряд згустку: n — найменший власний вектор блоку (e3,e4,e5), знак піднято BFS (RP² → S²), Q = Σ тілесних кутів / 4π; λ_eff з площі n_z < 0"""
    L = P.shape[0]; w, v = np.linalg.eigh(P[..., 2:5, 2:5]); n = v[..., :, 0].copy()
    from collections import deque
    seen = np.zeros((L, L), bool); seen[0, 0] = True
    if n[0, 0, 0] < 0: n[0, 0] *= -1
    q = deque([(0, 0)])
    while q:
        x, y = q.popleft()
        for dx, dy in PH.NB:
            u = ((x + dx) % L, (y + dy) % L)
            if not seen[u]:
                if np.dot(n[u], n[x, y]) < 0: n[u] *= -1
                seen[u] = True; q.append(u)
    sh = lambda A, d: np.roll(A, (-d[0], -d[1]), axis=(0, 1)); Om = 0.0
    for t in [((0, 0), (1, 0), (0, 1)), ((1, 0), (1, 1), (0, 1))]:
        a, b, c = [sh(n, d) for d in t]
        num = np.einsum("...i,...i", a, np.cross(b, c)); den = 1 + np.einsum("...i,...i", a, b) + np.einsum("...i,...i", b, c) + np.einsum("...i,...i", c, a)
        Om += (2*np.arctan2(num, den)).sum()
    return float(Om/(4*np.pi)), float(np.sqrt((n[..., 0] < 0).sum()*np.sqrt(3)/2/np.pi))

def clusters(det, X, Y, L):
    pts = [(X[np.unravel_index(k, X.shape)], Y[np.unravel_index(k, X.shape)]) for k in np.flatnonzero((det < 0).any(0))]
    cl = []
    for p in pts:
        for c in cl:
            if min(abs(p[0] - c[0]) % L, L - abs(p[0] - c[0]) % L) < 4 and abs(p[1] - c[1]) < 4: break
        else: cl.append(p)
    return [(round(float(a), 1), round(float(b), 1)) for a, b in cl]

def edens(P, V):
    P2 = P @ P; t2 = np.trace(P2, axis1=-2, axis2=-1)
    e = 0.5*np.sum(V**2, axis=(-2, -1)) - t2/2 + t2**2/4 + 0.5/4*np.trace(P2 @ P2, axis1=-2, axis2=-1)
    for d in [(1, 0), (0, 1), (1, -1)]:
        e += (1/3)*np.sum((np.roll(P, (-d[0], -d[1]), axis=(0, 1)) - P)**2, axis=(-2, -1))
    D = np.diag([U_]*3 + [V_]*2); D2 = D @ D; t20 = np.trace(D2)
    return e - (-t20/2 + t20**2/4 + 0.5/4*np.trace(D2 @ D2))

def run(kind="lump", L=64, lam=4.0, d=16.0, T=200.0, dt=0.05, Tr=5.0, gam=0.0):
    P = PH.sym0(init(kind, L, lam, d)); V = np.zeros_like(P); a = np.full((L, L), -1.0); B0 = np.zeros((5, 5))
    X, Y = PH.coords(L); cx, cy = X.mean(), Y.mean(); r, _, _ = wrapd(X, Y, cx, cy, L)
    t = 0.0; rec = []
    while t < T - 1e-9:
        g = 1.0 if t < Tr else gam
        P, V = PH.evolve(P, V, a, 0.0, B0, dt, g); t += dt
        if abs(t - round(t)) < dt/2 and (round(t) % 5 == 0 or t < 6):
            det, ang = holonomy(P); e = edens(P, V); E = e.sum()
            nz2 = int((det < 0).sum()); Q = float(ang[det > 0].sum()/(2*np.pi))
            ep = np.clip(e, 0, None); R2 = float(np.sqrt((ep*r**2).sum()/max(ep.sum(), 1e-30)))
            loc = float(ep[r < 12].sum()/max(ep.sum(), 1e-30))
            cl = clusters(det, X, Y, L); Ql, lam_e = lumpQ(P) if kind == "lump" else (0.0, 0.0)
            rec.append(dict(t=round(t, 2), E=float(E), nZ2=nz2, ncl=len(cl), cl=cl[:6], Ql=Ql, lam=lam_e, loc=loc, emax=float(e.max())))
            print(rec[-1], flush=True)
    name = f"matter98_{kind}_L{L}_lam{lam:g}_d{d:g}_g{gam:g}_Tr{Tr:g}"
    json.dump(rec, open(name + ".json", "w")); np.savez(name + ".npz", P=P, V=V)
    return name, rec

if __name__ == "__main__":
    kw = {k: (v if k == "kind" else (int(v) if k == "L" else float(v))) for k, v in (a.split("=") for a in sys.argv[1:])}
    run(**kw)
