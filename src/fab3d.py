"""fab3d: 3D Matrix (queue No. 60, step 1). Face-centred cubic lattice (FCC) on the 3-torus L×L×L: 12 neighbours per node,
all elementary cells are tetrahedra and octahedra with triangular faces, so the postulate "triangular cell" is kept.
Node (i, j, k), links — 6 types u ∈ {e1, e2, e3, e1−e2, e2−e3, e1−e3} from each node (id = t·n + node, tail = x, head = x + u).
A triangle is a triple of mutual neighbours (x, x+u, x+v), u, v, v−u ∈ neighbours; 8 triangles per node, each link in 4 triangles (in 2D — in 2).
Cartesian embedding: a1 = (0,1,1)/√2, a2 = (1,0,1)/√2, a3 = (1,1,0)/√2 (nearest distance 1).
Interface — as simNlib.Fabric (tail, head, E1..S3, TV, tri_links, holonomy, ww), so sim34d.M34 and stall43.Sys work unchanged.
Local law: P = (I + A)/(1 + 8), K = Pⁿ via FFT over nodes (6×6 block per wave vector), checked against a sparse Pⁿ."""
import numpy as np, scipy.sparse as sp
import simNlib as N

U6 = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, -1, 0), (0, 1, -1), (1, 0, -1)])
A3 = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])/np.sqrt(2)          # rows — a1, a2, a3

class Fabric3D(N.Fabric):
    def __init__(s, L):
        s.L = L; n = L**3; s.n = n
        I = np.indices((L, L, L)).reshape(3, -1).T                      # node coordinates
        idx = lambda c: ((c[..., 0] % L)*L + (c[..., 1] % L))*L + (c[..., 2] % L)
        s.tail = np.concatenate([np.arange(n)]*6); s.head = np.concatenate([idx(I + u) for u in U6]); s.m = 6*n
        s.ijk = I
        nbv = np.concatenate([U6, -U6]); key = {tuple(v): k for k, v in enumerate(nbv)}
        def link(x, u):                                                  # link x → x+u: (id, sign)
            for t, w in enumerate(U6):
                if tuple(u) == tuple(w): return t*n + x, np.ones_like(x)
                if tuple(u) == tuple(-w): return t*n + idx(I[x] + u), -np.ones_like(x)
            raise KeyError(u)
        tri = set(); TV = []; T = []
        x = np.arange(n)
        pairs = [(u, v) for a, u in enumerate(nbv) for b, v in enumerate(nbv) if a < b and tuple(v - u) in key]
        for u, v in pairs:                                               # triangle (x, x+u, x+v), circuit x → x+u → x+v → x
            q = idx(I + u); r = idx(I + v)
            e1, s1 = link(x, u); e2, s2 = link(q, v - u); e3, s3 = link(r, -v)
            for j in range(n):
                k_ = tuple(sorted((j, q[j], r[j])))
                if k_ in tri: continue
                tri.add(k_); TV.append((j, q[j], r[j])); T.append((e1[j], s1[j], e2[j], s2[j], e3[j], s3[j]))
        T = np.array(T); s.E1, s.S1, s.E2, s.S2, s.E3, s.S3 = T.T; s.TV = np.array(TV); s.ntri = len(T)
        cnt = np.bincount(np.r_[s.E1, s.E2, s.E3], None, s.m)
        assert s.ntri == 8*n and (cnt == 4).all(), (s.ntri, np.unique(cnt))
        s.pos = I @ A3                                                   # Cartesian node coordinates
        s.xy = I                                                         # (compatibility: integer coordinates)
    def wilson(s, z): return (0.0, 0.0)
    def disp(s):
        """Cartesian vector of the link tail → head (u in the basis a)"""
        return np.repeat(U6 @ A3, s.n, axis=0)

def local_P(f):
    E3 = np.stack([f.E1, f.E2, f.E3], 1); rows = []; cols = []
    for a, b in ((0, 1), (1, 2), (0, 2)): rows += [E3[:, a], E3[:, b]]; cols += [E3[:, b], E3[:, a]]
    A = sp.csr_matrix((np.ones(sum(len(r) for r in rows)), (np.concatenate(rows), np.concatenate(cols))), shape=(f.m, f.m))
    A.data[:] = 1.0; deg = np.asarray(A.sum(1)).ravel(); assert (deg == 8).all()
    P = ((sp.identity(f.m) + A)/9.0).tocsr(); assert np.allclose(np.asarray(P.sum(0)).ravel(), 1.0)
    return P

def fft_K(f, P, n):
    """K = Pⁿ as a function: a 6×6 block per wave vector (P is translation invariant)."""
    L = f.L; nn = f.n; W = np.zeros((6, 6, L, L, L))
    Pc = P.tocsr()
    for t in range(6):                                                   # row of a link of type t at node 0
        row = Pc.getrow(t*nn);
        for c, w in zip(row.indices, row.data):
            t2, y = divmod(int(c), nn); W[t, t2][np.unravel_index(y, (L, L, L))] += w
    # out_t(x) = Σ w(t,t',d) src_t'(x+d) → in Fourier space (numpy): factor Σ w e^{+ik·d} = conj(fft(w)) for real w
    Pk = np.conj(np.fft.fftn(W, axes=(2, 3, 4))).transpose(2, 3, 4, 0, 1)   # (L,L,L,6,6)
    Kk = np.linalg.matrix_power(Pk, n)
    def K(src):
        S = np.fft.fftn(src.reshape(6, L, L, L), axes=(1, 2, 3)).transpose(1, 2, 3, 0)
        out = np.einsum("...ab,...b->...a", Kk, S).transpose(3, 0, 1, 2)
        return np.real(np.fft.ifftn(out, axes=(1, 2, 3))).reshape(-1)
    return K

def tau3d(L, ell, seed):
    """smooth random periodic τ, isotropic in the Cartesian FCC embedding"""
    rng = np.random.default_rng(700 + seed); w = rng.standard_normal((L, L, L))
    k = np.fft.fftfreq(L)*2*np.pi; KI = np.stack(np.meshgrid(k, k, k, indexing="ij"), -1)
    G = np.linalg.inv(A3) @ np.linalg.inv(A3).T                           # k_cart = A⁻¹ k_idx (since x_idx·k_idx = x_cart·k_cart)
    K2 = np.einsum("...a,ab,...b->...", KI, G, KI)
    t = np.real(np.fft.ifftn(np.fft.fftn(w)*np.exp(-K2*ell**2/2))); return t.ravel()/t.std()

def vortex_lines(L, k, seed):
    """(109): k pairs of pre-time vortex lines along the third index axis, charges +1/−1; positions (a, b) in the plane of the first two indices,
    shifted by 1/3 so that no node lies on a line (rng 800 + seed)."""
    rng = np.random.default_rng(800 + seed); c = rng.integers(0, L, (2*k, 2)) + 1/3
    return [(float(a), float(b), 1 if i % 2 == 0 else -1) for i, (a, b) in enumerate(c)]

def vortex_lines_dtheta(f, L, cores, M=3):
    """increment of the cyclic pre-time θ along the link tail→head: Σ q·Δarg around each line (in the index plane i, j) and its images on the torus"""
    pt = f.ijk[f.tail, :2].astype(float); u = np.repeat(U6[:, :2], f.n, axis=0).astype(float); ph = pt + u
    out = np.zeros(f.m)
    for (a, b, q) in cores:
        for m in range(-M, M + 1):
            for n in range(-M, M + 1):
                ca, cb = a + m*L, b + n*L
                d_ = np.arctan2(ph[:, 1] - cb, ph[:, 0] - ca) - np.arctan2(pt[:, 1] - cb, pt[:, 0] - ca)
                out += q*((d_ + np.pi) % (2*np.pi) - np.pi)
    return out
