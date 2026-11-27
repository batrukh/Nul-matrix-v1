"""simNlib: bubble with its own dynamic boundary coexisting with the Matrix.
Torus fabric (triangular L x L lattice, complex link weights z_e) + node order parameter psi
(model A: psi=-1 Matrix, psi=+1 bubble phase, tilt h makes the bubble lower). chi = ((1+psi)/2)^2.
F = sum_e [g|z|^2/2 + mu|z|^4/4] + breaking - beta sum_tri Re(z1 z2 z3)
  + sum_nodes [(psi^2-1)^2/4 - h psi] + kappa/2 sum_e (psi_a-psi_b)^2
breaking: mode 'link' = sum_e eps0 chi_a chi_b |z|(|z|-Re z) (gauge-dependent);
          mode 'tri'  = sum_tri eps0 mean(chi) (|P|-Re P), P = z1 z2 z3 (gauge-invariant).
Re/Im parts of z and psi follow minus the gradient of F (one gradient flow)."""
import numpy as np
from scipy.integrate import solve_ivp
g, mu = 1.0, 1.0

class Fabric:
    def __init__(self, L):
        self.L = L; n = L*L; self.n = n
        idx = lambda x, y: (x % L)*L + (y % L)
        tail, head, eid = [], [], {}
        def edge(u, v):
            key = (min(u, v), max(u, v))
            if key not in eid:
                eid[key] = len(tail); tail.append(u); head.append(v)
            e = eid[key]; return e, (1 if tail[e] == u else -1)
        T = []; TV = []
        for x in range(L):
            for y in range(L):
                a, b, c, d = idx(x, y), idx(x+1, y), idx(x, y+1), idx(x+1, y+1)
                for p, q, r in ((a, b, c), (b, d, c)):      # counter-clockwise
                    T.append([*edge(p, q), *edge(q, r), *edge(r, p)]); TV.append((p, q, r))
        self.tail, self.head = np.array(tail), np.array(head); self.m = len(tail)
        T = np.array(T); self.E1, self.S1, self.E2, self.S2, self.E3, self.S3 = T.T
        self.ntri = len(T); self.TV = np.array(TV)
        self.xy = np.array([(i//L, i % L) for i in range(n)])
        self.wx = [edge(idx(x, 0), idx(x+1, 0)) for x in range(L)]   # Wilson loop along x
        self.wy = [edge(idx(0, y), idx(0, y+1)) for y in range(L)]   # Wilson loop along y

    def get(self, z, E, S): return np.where(S > 0, z[E], np.conj(z[E]))

    def tri_links(self, z):
        return self.get(z, self.E1, self.S1), self.get(z, self.E2, self.S2), self.get(z, self.E3, self.S3)

    def holonomy(self, z):
        l1, l2, l3 = self.tri_links(z); return np.angle(l1*l2*l3)

    def ww(self, z):
        """(W@W) restricted to links, in stored orientation."""
        l1, l2, l3 = self.tri_links(z)
        c1, c2, c3 = np.conj(l3*l2), np.conj(l1*l3), np.conj(l2*l1)
        out = np.zeros(self.m, complex)
        for E, S, c in ((self.E1, self.S1, c1), (self.E2, self.S2, c2), (self.E3, self.S3, c3)):
            v = np.where(S > 0, c, np.conj(c))
            out += np.bincount(E, v.real, self.m) + 1j*np.bincount(E, v.imag, self.m)
        return out

    def wilson(self, z):
        w1 = np.prod([z[e] if s > 0 else np.conj(z[e]) for e, s in self.wx])
        w2 = np.prod([z[e] if s > 0 else np.conj(z[e]) for e, s in self.wy])
        return np.angle(w1), np.angle(w2)

def wplus(beta): return (2*beta + np.sqrt(4*beta**2 - 4*g*mu))/(2*mu)

def matrix_state(fab, beta, k, rng, noise=0.0, gauge_amp=np.pi):
    """Uniform Matrix with total flux k (twisted Landau gauge on the torus) + random gauge of amplitude gauge_amp."""
    L = fab.L; phi_cell = 2*np.pi*k/L**2
    th = np.zeros(fab.m)
    for e in range(fab.m):
        (x1, y1), (x2, y2) = fab.xy[fab.tail[e]], fab.xy[fab.head[e]]
        dx = (x2 - x1 + L//2) % L - L//2; dy = (y2 - y1 + L//2) % L - L//2
        th[e] = -phi_cell*y1*dx
        if dy == 1 and y1 == L-1: th[e] += phi_cell*L*x2
        if dy == -1 and y2 == L-1: th[e] -= phi_cell*L*x1
    gauge = rng.uniform(-gauge_amp, gauge_amp, fab.n)
    th += gauge[fab.tail] - gauge[fab.head]
    return wplus(beta)*np.exp(1j*th)*(1 + noise*rng.standard_normal(fab.m))

def disk(fab, R, cx=None, cy=None):
    L = fab.L; cx = L//2 if cx is None else cx; cy = L//2 if cy is None else cy
    dx = (fab.xy[:, 0]-cx + L//2) % L - L//2; dy = (fab.xy[:, 1]-cy + L//2) % L - L//2
    X, Y = dx + dy/2, dy*np.sqrt(3)/2
    return np.sqrt(X**2 + Y**2)

# addition to simNlib.py
def sparsity(fab):
    """Jacobian sparsity: within each triangle its 3 edges (Re, Im) and 3 nodes (psi) are mutually coupled."""
    import scipy.sparse as sp
    m, n = fab.m, fab.n; rows, cols = [], []
    for t in range(fab.ntri):
        es = [fab.E1[t], fab.E2[t], fab.E3[t]]; ns = list(fab.TV[t])
        v = es + [m + e for e in es] + [2*m + q for q in ns]
        for i in v:
            for j in v: rows.append(i); cols.append(j)
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(2*m + n, 2*m + n)).astype(bool).astype(float)
