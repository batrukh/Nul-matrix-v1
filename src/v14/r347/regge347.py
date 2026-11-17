"""(347) постулат-кандидат 3.9 «тяжіння як баланс кутів»: дефіцит кута у вузлі δ_v = 8πG·E_v, довжини ниток підлаштовуються.
Трикутна ґратка-диск радіуса R. Довжини l_ij = l0·exp((u_i+u_j)/2) (один масштаб на вузол — дискретна конформна форма, вибір калібру).
Ньютон: 2π − Σθ_v(u) = δ_v у внутрішніх вузлах, u = 0 на краю. Далі: розгортка на площину з розрізом уздовж +x (для світла),
котангенсний лапласіан із довжин (хвиля), кола довкола центру (C(r)/2πr)."""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl
def disk(R):
    a1, a2 = np.array([1.0, 0.0]), np.array([0.5, np.sqrt(3)/2]); idx = {}; P = []
    for a in range(-R-2, R+3):
        for b in range(-R-2, R+3):
            p = a*a1 + b*a2
            if np.hypot(*p) <= R + 1e-9: idx[(a, b)] = len(P); P.append(p)
    P = np.array(P); T = []
    for (a, b), i in idx.items():
        for t in (((a+1, b), (a, b+1)), ((a+1, b), (a+1, b-1))):
            j, k = idx.get(t[0]), idx.get(t[1])
            if j is not None and k is not None: T.append((i, j, k))
    T = np.array(T); E = np.unique(np.sort(np.r_[T[:, [0, 1]], T[:, [1, 2]], T[:, [0, 2]]], 1), axis=0)
    deg = np.bincount(E.ravel(), minlength=len(P)); ntri = np.bincount(T.ravel(), minlength=len(P))
    inner = ntri == 6
    return P, T, E, inner
def angles(L0, u, T):
    i, j, k = T.T
    a = L0[0]*np.exp((u[j] + u[k])/2); b = L0[1]*np.exp((u[i] + u[k])/2); c = L0[2]*np.exp((u[i] + u[j])/2)   # a — навпроти i
    ci = np.clip((b*b + c*c - a*a)/(2*b*c), -1, 1); cj = np.clip((a*a + c*c - b*b)/(2*a*c), -1, 1); ck = np.clip((a*a + b*b - c*c)/(2*a*b), -1, 1)
    return np.arccos(ci), np.arccos(cj), np.arccos(ck), (a, b, c)
def tri_L0(P, T):
    i, j, k = T.T
    return np.hypot(*(P[j] - P[k]).T), np.hypot(*(P[i] - P[k]).T), np.hypot(*(P[i] - P[j]).T)
def angsum(L0, u, T, n):
    ti, tj, tk, _ = angles(L0, u, T)
    return np.bincount(T[:, 0], ti, n) + np.bincount(T[:, 1], tj, n) + np.bincount(T[:, 2], tk, n)
def solve(P, T, inner, delta, tol=1e-11, it=60, verbose=False):
    n = len(P); L0 = tri_L0(P, T); u = np.zeros(n); I = np.where(inner)[0]; pos = -np.ones(n, int); pos[I] = np.arange(len(I))
    for k in range(it):
        F = (2*np.pi - angsum(L0, u, T, n) - delta)[I]
        err = np.abs(F).max()
        if verbose: print(k, err, flush=True)
        if err < tol: break
        # якобіан: ∂θ/∂u скінченними різницями по трикутнику (3×3), збірка
        h = 1e-7; rows, cols, vals = [], [], []
        th0 = np.array(angles(L0, u, T)[:3])
        for m in range(3):
            du = np.zeros(n); uu = u.copy()
            # збурюємо m-ту вершину кожного трикутника окремо: рахуємо по трикутниках напряму
            i, j, k_ = T.T
            ue = [u[i].copy(), u[j].copy(), u[k_].copy()]; ue[m] = ue[m] + h
            a = L0[0]*np.exp((ue[1] + ue[2])/2); b = L0[1]*np.exp((ue[0] + ue[2])/2); c = L0[2]*np.exp((ue[0] + ue[1])/2)
            th = np.array([np.arccos(np.clip((b*b + c*c - a*a)/(2*b*c), -1, 1)), np.arccos(np.clip((a*a + c*c - b*b)/(2*a*c), -1, 1)), np.arccos(np.clip((a*a + b*b - c*c)/(2*a*b), -1, 1))])
            d = (th - th0)/h
            for q in range(3):
                rows.append(T[:, q]); cols.append(T[:, m]); vals.append(-d[q])     # F = 2π − Σθ − δ
        rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
        ok = (pos[rows] >= 0) & (pos[cols] >= 0)
        J = sp.csr_matrix((vals[ok], (pos[rows[ok]], pos[cols[ok]])), shape=(len(I), len(I)))
        du = spl.spsolve(J.tocsc(), -F); lam = 1.0
        u[I] += lam*du
    return u, err, k
def layout(P, T, u, allow=None, start=(-3.0, 0.3)):
    """Розгортка на площину BFS по трикутниках (координати свої в кожного трикутника), не переходячи через розріз:
    ребра ряду y = 0 при x ≥ 0 (позаду маси). Вузол отримує координати першого трикутника, що його поставив (вузли на розрізі — з нижнього або верхнього боку)."""
    from collections import deque
    n = len(P); L0 = tri_L0(P, T); ti, tj, tk, (a, b, c) = angles(L0, u, T); nt = len(T)
    edge_tris = {}
    for t, (i, j, k) in enumerate(T):
        for e in ((i, j), (j, k), (i, k)): edge_tris.setdefault(tuple(sorted(e)), []).append(t)
    def is_cut(e):
        p, q = P[e[0]], P[e[1]]
        return abs(p[1]) < 1e-9 and abs(q[1]) < 1e-9 and min(p[0], q[0]) > -1e-9
    def L(t, x, y):
        i, j, k = T[t]; s = {x, y}
        if s == {j, k}: return a[t]
        if s == {i, k}: return b[t]
        return c[t]
    TQ = {}
    t0 = int(np.argmin(np.hypot(*(P[T].mean(1) - np.array(start)).T)))
    ok = np.ones(nt, bool) if allow is None else allow   # дозволені трикутники (напр. лише пласка зовнішня частина)
    i, j, k = T[t0]; qi = np.array([0.0, 0.0]); qj = np.array([c[t0], 0.0]); qk = b[t0]*np.array([np.cos(ti[t0]), np.sin(ti[t0])])
    o1 = (P[j]-P[i])[0]*(P[k]-P[i])[1] - (P[j]-P[i])[1]*(P[k]-P[i])[0]
    if o1 < 0: qk[1] *= -1
    TQ[t0] = {i: qi, j: qj, k: qk}
    Q = np.full((n, 2), np.nan)
    for v_, q_ in TQ[t0].items(): Q[v_] = q_
    dq = deque([t0])
    while dq:
        t = dq.popleft(); tri = list(T[t])
        for e in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])):
            es = tuple(sorted(e))
            if is_cut(es): continue
            for t2 in edge_tris[es]:
                if t2 in TQ or not ok[t2]: continue
                x, y = es; z = [v for v in T[t2] if v not in es][0]; other = [v for v in T[t] if v not in es][0]
                Qx, Qy = TQ[t][x], TQ[t][y]
                lx, ly = L(t2, x, z), L(t2, y, z); d = Qy - Qx; dl = np.hypot(*d); e1 = d/dl; e2 = np.array([-e1[1], e1[0]])
                s_ = (lx*lx - ly*ly + dl*dl)/(2*dl); hgt = np.sqrt(max(lx*lx - s_*s_, 0))
                side = np.sign(np.dot(TQ[t][other] - Qx, e2)) or 1.0
                Qz = Qx + s_*e1 - side*hgt*e2
                TQ[t2] = {x: Qx, y: Qy, z: Qz}
                if np.isnan(Q[z]).any(): Q[z] = Qz
                dq.append(t2)
    return Q
def cot_lap(P, T, u):
    n = len(P); L0 = tri_L0(P, T); ti, tj, tk, (a, b, c) = angles(L0, u, T)
    i, j, k = T.T; area = 0.5*b*c*np.sin(ti)
    w = {}; r, cc, v = [], [], []
    for (p, q, th) in ((j, k, ti), (i, k, tj), (i, j, tk)):
        ct = 0.5/np.tan(th); r += [p, q]; cc += [q, p]; v += [ct, ct]
    W = sp.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(cc))), shape=(n, n))
    K = sp.diags(np.asarray(W.sum(1)).ravel()) - W
    M = np.bincount(T.ravel(), np.repeat(area/3, 3), n)
    return K, M, area
