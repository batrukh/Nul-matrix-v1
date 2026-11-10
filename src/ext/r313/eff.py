"""(313) Ефективна модель непарного кванта (313): дійсне 2-компонентне поле χ (площина e2,e3) на трикутній сітці
поверх пари вихорів Аліси, посаджених як у (305): директор n = R(θ/2)e2, ланки rot(−(1−f)Δθ/2), f = exp(−(r/r0)²).
D = t·Σ_ланки |χ_x − U χ_{x+d}|² + Σ_x χᵀ M_x χ,  M = m2²(r)·nnᵀ + m3²(1−nnᵀ),  m2² = m∞²(1 − h·exp(−(r/rc)²)).
Квант бозонний (ω² — власні значення D). MODE: odd — як вище; even — контроль без Z₂: ланки = 1, n = e2 скрізь."""
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh
DIRS = [(1, 0), (0, 1), (-1, 1)]
def rot(a):
    c, s = np.cos(a), np.sin(a); M = np.empty(a.shape + (2, 2)); M[..., 0, 0] = c; M[..., 0, 1] = -s; M[..., 1, 0] = s; M[..., 1, 1] = c; return M
def config(L, R, r0, phi, c=None, single=False):
    x, y = np.meshgrid(np.arange(L), np.arange(L), indexing='ij'); S = np.stack([x + 0.5*y, np.sqrt(3)/2*y], -1)
    c = np.array([0.5*L + 0.25*L + 0.5, 0.5*np.sqrt(3)/2*L + np.sqrt(3)/6]) if c is None else c   # центр трикутника
    e = np.array([np.cos(phi), np.sin(phi)]); v1, v2 = c + R*e, c - R*e
    if single: th = lambda Q: np.arctan2(Q[..., 1] - v1[1], Q[..., 0] - v1[0])
    else: th = lambda Q: np.arctan2(Q[..., 1] - v1[1], Q[..., 0] - v1[0]) - np.arctan2(Q[..., 1] - v2[1], Q[..., 0] - v2[0])
    t = th(S); n = np.stack([np.cos(t/2), np.sin(t/2)], -1)
    dist = lambda Q: np.minimum(np.hypot(Q[..., 0] - v1[0], Q[..., 1] - v1[1]), 1e9 if single else np.hypot(Q[..., 0] - v2[0], Q[..., 1] - v2[1]))
    U = np.empty((3, L, L, 2, 2))
    for m, d in enumerate(DIRS):
        S2 = S + np.array([d[0] + 0.5*d[1], np.sqrt(3)/2*d[1]]); M = 0.5*(S + S2)
        dth = np.angle(np.exp(1j*(th(S2) - t))); f = np.exp(-(dist(M)/r0)**2)
        U[m] = rot(-(1 - f)*dth/2)
    return n, U, dist(S), (v1, v2)
def build(L, n, U, r, minf=0.333, m3=0.6, h=0.5, rc=3.0, mode='odd', t=1.0):
    N = L*L; idx = np.arange(N).reshape(L, L)
    if mode == 'even': n = np.broadcast_to(np.array([1.0, 0.0]), n.shape); U = np.broadcast_to(np.eye(2), U.shape)
    nn = np.einsum('...i,...j', n, n); m2 = minf**2*(1 - h*np.exp(-(r/rc)**2))
    M = m2[..., None, None]*nn + m3**2*(np.eye(2) - nn) + 6*t*np.eye(2)          # 6 — діагональ від шести сусідів
    rows, cols, vals = [], [], []
    def blk(ia, ib, B):
        rr = ia[:, None, None]*2 + np.arange(2)[None, :, None] + 0*np.arange(2)[None, None, :]
        cc = ib[:, None, None]*2 + np.arange(2)[None, None, :] + 0*np.arange(2)[None, :, None]
        rows.append(rr.ravel()); cols.append(cc.ravel()); vals.append(B.reshape(-1, 2, 2).ravel())
    a = idx.ravel(); blk(a, a, M)
    for m, d in enumerate(DIRS):
        b = np.roll(idx, (-d[0], -d[1]), axis=(0, 1)).ravel()
        blk(a, b, -t*U[m]); blk(b, a, -t*np.swapaxes(U[m], -1, -2))
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2*N, 2*N))
def low(D, k):
    w, v = eigsh(D, k=k, sigma=0.0, which='LM'); o = np.argsort(w); return w[o], v[:, o]
