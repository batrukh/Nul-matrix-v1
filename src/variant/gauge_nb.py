"""gauge_nb: швидке (numba) ядро для gauge91.Gauge.forces/step і phi90.dV у кроці Gauge.
Та сама фізика, той самий порядок операцій з точністю до округлення. Підключення: import gauge_nb; gauge_nb.install()."""
import numpy as np, numba as nb

@nb.njit(inline='always')
def mm(A, B, C):            # C = A @ B (5×5)
    for i in range(5):
        for j in range(5):
            s = 0.0
            for k in range(5): s += A[i, k]*B[k, j]
            C[i, j] = s
@nb.njit(inline='always')
def mmT(A, B, C):           # C = A @ Bᵀ
    for i in range(5):
        for j in range(5):
            s = 0.0
            for k in range(5): s += A[i, k]*B[j, k]
            C[i, j] = s
@nb.njit(inline='always')
def mTm(A, B, C):           # C = Aᵀ @ B
    for i in range(5):
        for j in range(5):
            s = 0.0
            for k in range(5): s += A[k, i]*B[k, j]
            C[i, j] = s

@nb.njit(cache=True)
def forces_nb(P, U, a, b, kb, wt, ka, wl):
    L = P.shape[0]
    FP = np.zeros_like(P); R = np.zeros_like(U)
    T1 = np.empty((5, 5)); T2 = np.empty((5, 5)); X = np.empty((5, 5)); D = np.empty((5, 5)); H = np.empty((5, 5)); P2 = np.empty((5, 5)); P3 = np.empty((5, 5))
    di = (1, 0, -1); dj = (0, 1, 1)
    for i in range(L):
        for j in range(L):
            p = P[i, j]
            mm(p, p, P2); mm(P2, p, P3)
            t2 = 0.0
            for k in range(5): t2 += P2[k, k]
            aa = a[i, j]
            for r in range(5):
                for c in range(5):
                    FP[i, j, r, c] -= aa*p[r, c] - b*P2[r, c] + t2*p[r, c] + 0.5*P3[r, c]
            for m in range(3):
                i2 = (i + di[m]) % L; j2 = (j + dj[m]) % L
                Um = U[m, i, j]
                mm(Um, P[i2, j2], T1); mmT(T1, Um, X)
                for r in range(5):
                    for c in range(5): D[r, c] = X[r, c] - p[r, c]
                mTm(Um, D, T1); mm(T1, Um, T2)
                mm(D, X, H)
                for r in range(5):
                    for c in range(5):
                        FP[i, j, r, c] += (2.0/3.0)*D[r, c]
                        FP[i2, j2, r, c] -= (2.0/3.0)*T2[r, c]
                        R[m, i, j, r, c] += (4.0/3.0)*H[r, c]
            # трикутник вгору: H = U0(a) U2(a+e0) U1(a)ᵀ
            ie = (i + 1) % L
            bu = kb*wt[0, i, j]
            mm(U[0, i, j], U[2, ie, j], T1); mmT(T1, U[1, i, j], H)
            A = U[0, i, j]
            mTm(A, H, T1)            # Aᵀ H ... need Aᵀ Hᵀ A
            for r in range(5):
                for c in range(5):
                    R[0, i, j, r, c] -= bu*H[c, r]
                    R[1, i, j, r, c] -= bu*H[r, c]
            # Aᵀ Hᵀ A
            for r in range(5):
                for c in range(5):
                    s = 0.0
                    for k in range(5): s += A[k, r]*H[c, k]   # (Aᵀ Hᵀ)[r,c] = Σ_k A[k,r] H[c,k]
                    T1[r, c] = s
            mm(T1, A, T2)
            for r in range(5):
                for c in range(5): R[2, ie, j, r, c] -= bu*T2[r, c]
            # трикутник вниз (база a): A = U1(a+e0), B = U0(a+e1), C = U2(a+e0); H = A Bᵀ Cᵀ
            jn = (j + 1) % L
            bd = kb*wt[1, i, j]
            Aq = U[1, ie, j]; Bq = U[0, i, jn]; Cq = U[2, ie, j]
            mmT(Aq, Bq, T1); mmT(T1, Cq, H)
            # Cᵀ H C
            mTm(Cq, H, T1); mm(T1, Cq, T2)
            for r in range(5):
                for c in range(5):
                    R[1, ie, j, r, c] -= bd*H[c, r]
                    R[0, i, jn, r, c] -= bd*T2[r, c]
                    R[2, ie, j, r, c] -= bd*H[r, c]
    # sym0(FP), FE = −2/(ka wl) PA(R)
    FE = np.empty_like(U)
    for i in range(L):
        for j in range(L):
            tr = 0.0
            for k in range(5): tr += FP[i, j, k, k]
            for r in range(5):
                for c in range(r, 5):
                    v = 0.5*(FP[i, j, r, c] + FP[i, j, c, r])
                    FP[i, j, r, c] = v; FP[i, j, c, r] = v
                FP[i, j, r, r] -= tr/5.0
            for m in range(3):
                f = -2.0/(ka*wl[m, i, j])
                for r in range(5):
                    for c in range(5):
                        FE[m, i, j, r, c] = f*0.5*(R[m, i, j, r, c] - R[m, i, j, c, r])
    return FP, FE

@nb.njit(cache=True)
def cayley_mul_nb(E, U, dt):
    """U_new = (I − dt E/2)⁻¹ (I + dt E/2) U, вузол за вузлом (Гаусс з частковим вибором, 5×5)"""
    out = np.empty_like(U)
    M = np.empty((5, 5)); Y = np.empty((5, 5))
    for m in range(U.shape[0]):
        for i in range(U.shape[1]):
            for j in range(U.shape[2]):
                for r in range(5):
                    for c in range(5):
                        x = 0.5*dt*E[m, i, j, r, c]
                        M[r, c] = (1.0 if r == c else 0.0) - x
                    # права частина: (I + X/2) U
                    for c in range(5):
                        s = U[m, i, j, r, c]
                        for k in range(5): s += 0.5*dt*E[m, i, j, r, k]*U[m, i, j, k, c]
                        Y[r, c] = s
                for k in range(5):
                    p = k; best = abs(M[k, k])
                    for r in range(k + 1, 5):
                        if abs(M[r, k]) > best: best = abs(M[r, k]); p = r
                    if p != k:
                        for c in range(5):
                            t = M[k, c]; M[k, c] = M[p, c]; M[p, c] = t
                            t = Y[k, c]; Y[k, c] = Y[p, c]; Y[p, c] = t
                    for r in range(k + 1, 5):
                        f = M[r, k]/M[k, k]
                        for c in range(k, 5): M[r, c] -= f*M[k, c]
                        for c in range(5): Y[r, c] -= f*Y[k, c]
                for r in range(4, -1, -1):
                    for c in range(5):
                        s = Y[r, c]
                        for k in range(r + 1, 5): s -= M[r, k]*out[m, i, j, k, c]
                        out[m, i, j, r, c] = s/M[r, r]
    return out

def install():
    import gauge91 as G, phi90 as PH
    if getattr(G.Gauge, '_nb', False): return
    def forces(s, P, U):
        return forces_nb(np.ascontiguousarray(P), np.ascontiguousarray(U), np.ascontiguousarray(s.a, dtype=np.float64), float(s.b), float(s.kb),
                         np.ascontiguousarray(s.wt, dtype=np.float64), float(s.ka), np.ascontiguousarray(s.wl, dtype=np.float64))
    def step(s, P, Pi, U, E, dt, gam=0.0):
        FP, FE = s.forces(P, U)
        Pi = Pi + 0.5*dt*(FP - gam*Pi); E = E + 0.5*dt*(FE - gam*E)
        P = PH.sym0(P + dt*Pi); U = cayley_mul_nb(np.ascontiguousarray(E), np.ascontiguousarray(U), dt)
        FP, FE = s.forces(P, U)
        Pi = (Pi + 0.5*dt*FP)/(1 + 0.5*dt*gam); E = (E + 0.5*dt*FE)/(1 + 0.5*dt*gam)
        return P, Pi, U, E
    G.Gauge.forces = forces; G.Gauge.step = step; G.Gauge._nb = True
