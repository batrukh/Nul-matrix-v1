"""(304) Ферміон на соті поверх вихору бази (r294, κa 1, κb 2, b 0,1, L96).
Сота = центри трикутників трикутної сітки: A(a) = up {a, a+e0, a+e1}, B(a) = down {a, a+e1, a+e2}.
Ферміон ψ — вектор SO(5) (новий постулат), рамка трикутника = рамка базового вузла a.
Стрибки A(a)–B(a): I;  A(a)–B(a+e0): U0(a);  A(a)–B(a−e2): U2(a−e2)ᵀ   (калібрувально коваріантно).
Маса: +g·Φ_tri на A, −g·Φ_tri на B (Φ-маса Семенова), Φ_tri — середнє трьох вершин, перенесене в рамку a.
H = −t Σ (ψ_A† T ψ_B + h.c.) + g Σ_A ψ†Φψ − g Σ_B ψ†Φψ."""
import numpy as np, scipy.sparse as sp, sys
sys.path.insert(0, '/mnt/project-files/model/s52/r268/code')
T_ = lambda A: np.swapaxes(A, -1, -2)
def sh(A, d): return np.roll(A, (-d[0], -d[1]), axis=(0, 1))
E0, E1, E2 = (1, 0), (0, 1), (-1, 1)

def build(P, U, g=1.0, t=1.0):
    L = P.shape[0]; N = L*L; I5 = np.eye(5)
    idx = np.arange(N).reshape(L, L)
    PhA = (P + U[0] @ sh(P, E0) @ T_(U[0]) + U[1] @ sh(P, E1) @ T_(U[1]))/3
    PhB = (P + U[1] @ sh(P, E1) @ T_(U[1]) + U[2] @ sh(P, E2) @ T_(U[2]))/3
    rows, cols, vals = [], [], []
    def blk(ia, ib, M):        # блоки 5×5: рядок ia (вузли), стовпець ib
        r = (ia[:, None, None]*5 + np.arange(5)[None, :, None]) + 0*np.arange(5)[None, None, :]
        c = (ib[:, None, None]*5 + np.arange(5)[None, None, :]) + 0*np.arange(5)[None, :, None]
        rows.append(r.ravel()); cols.append(c.ravel()); vals.append(M.reshape(-1, 5, 5).ravel())
    a = idx.ravel()
    blk(a, a, g*PhA); blk(a + N, a + N, -g*PhB)                   # A: 0..N-1, B: N..2N-1
    hops = [(a, np.broadcast_to(I5, (L, L, 5, 5))),
            (sh(idx, E0).ravel(), U[0]),
            (sh(idx, (1, -1)).ravel(), T_(sh(U[2], (1, -1))))]   # B(a−e2), U2(a−e2)ᵀ
    for ib, M in hops:
        blk(a, ib + N, -t*M); blk(ib + N, a, -t*T_(M))
    H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(10*N, 10*N))
    return H

def coordsAB(L):
    x, y = np.meshgrid(np.arange(L), np.arange(L), indexing='ij'); X, Y = x + 0.5*y, np.sqrt(3)/2*y
    return np.concatenate([(X + 0.5).ravel(), (X + 0.5).ravel() - 0.5 + 0.0]), np.concatenate([(Y + np.sqrt(3)/6).ravel(), (Y + np.sqrt(3)/3).ravel()])
