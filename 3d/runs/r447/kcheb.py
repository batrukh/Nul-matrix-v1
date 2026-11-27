"""(450-код, 21.11) Швидкий місцевий закон K = P^n на довільній ґратці: розклад Чебишова замість n множень.
P = diag(dg) + w·A (A — 0/1 матриця сусідства ниток, w = 1/(1+d_max)); спектр P у [λmin, 1], λmin знаходимо eigsh із запасом.
P^n x = Σ c_k T_k(Q) x, Q = (2P − (a+1)I)/(1 − a); c_k — інтерполяція Чебишова λ^n на [a, 1], обрізання хвоста < TOL.
Похибка ≤ TOL·|x| (перевірка test450.py), порядок вузлів — RCM для кешу."""
import numpy as np, numba as nb, scipy.sparse as sp, scipy.sparse.linalg as sl
from numpy.polynomial import chebyshev as C
from scipy.sparse.csgraph import reverse_cuthill_mckee

@nb.njit(cache=True)
def _cheb(ip, ix, dg, w, al, be, c, x):
    # Q y = al*(dg*y + w*A y) + be*y
    m = len(x); t0 = x.copy(); t1 = np.empty(m); t2 = np.empty(m); out = c[0]*x
    for i in range(m):
        acc = 0.0
        for p in range(ip[i], ip[i+1]): acc += t0[ix[p]]
        t1[i] = al*(dg[i]*t0[i] + w*acc) + be*t0[i]
    for i in range(m): out[i] += c[1]*t1[i]
    for k in range(2, len(c)):
        ck = c[k]
        for i in range(m):
            acc = 0.0
            for p in range(ip[i], ip[i+1]): acc += t1[ix[p]]
            v = 2.0*(al*(dg[i]*t1[i] + w*acc) + be*t1[i]) - t0[i]
            t2[i] = v; out[i] += ck*v
        t0, t1, t2 = t1, t2, t0
    return out

def make_K(P, n, tol=5e-14, margin=0.05):
    P = sp.csr_matrix(P); m = P.shape[0]
    dg = P.diagonal().copy(); O = P - sp.diags(dg); O.eliminate_zeros(); O = O.tocsr()
    ws = np.unique(O.data); assert len(ws) == 1, ws; w = float(ws[0])
    lmin = float(sl.eigsh(P, k=1, which="SA", return_eigenvectors=False, tol=1e-6)[0]); a = max(-1.0, lmin - margin)
    g = lambda u: ((u + 1)/2*(1 - a) + a)**n
    u = np.linspace(-1, 1, 40001); gu = g(u)
    for deg in range(10, 2*n + 1, 2):                                   # найменший степінь з похибкою функції ≤ tol на всьому [a, 1]
        c = C.chebinterpolate(g, deg)
        if np.abs(C.chebval(u, c) - gu).max() <= tol: break
    perm = reverse_cuthill_mckee(O, symmetric_mode=True); inv = np.empty_like(perm); inv[perm] = np.arange(m)
    Op = O[perm][:, perm].tocsr(); ip = Op.indptr.astype(np.int32); ix = Op.indices.astype(np.int32); dgp = np.ascontiguousarray(dg[perm])
    al = 2.0/(1 - a); be = -(a + 1)/(1 - a)
    def K(src):
        return _cheb(ip, ix, dgp, w, al, be, c, np.ascontiguousarray(np.asarray(src, float)[perm]))[inv]
    K.info = dict(lmin=lmin, a=a, deg=len(c), w=w); return K
