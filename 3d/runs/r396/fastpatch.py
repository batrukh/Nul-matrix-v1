"""fastpatch (17.11, (396)): прискорена сила для rq351f / rq383 / rq388 — та сама фізика, ті самі формули, той самий крок dt і RK4.
Що змінено лише в обчисленні: (1) сила петель — qf2.qforce2 (розгорнутий цикл, звірено з _qforce до 2e-14, 15.11);
(2) усі решта членів MQ.rhs + stall43.F + local60.F (до/після закону збереження) — один компільований прохід numba замість ~20 проходів numpy;
(3) закон збереження K = P^n через rfftn/irfftn (джерело дійсне) замість fftn/ifftn — математично тотожно.
Встановлення: import fastpatch; fastpatch.install()  — підміняє local60.Loc.F (до виклику run)."""
import os, sys, numpy as np, numba as nb
R = "/mnt/project-files/model/s52"; sys.path[:0] = [R + "/r351", R]
os.environ["NB"] = "0"; os.environ.setdefault("DELTA", "1e-6")
import qf2

@nb.njit(cache=True, fastmath=False)
def _pre(z, psi, ta, hb, lw, br, fcq, beta, cc, cs, kappa, wp, d2, h, lc, m0, n):
    """MQ.rhs (крім сили петель) + stall43.F; повертає dz (до закону збереження), fpsi, src = 2Re(z̄ dz), rho."""
    m = z.shape[0]; dz = np.empty(m, np.complex128); fpsi = np.zeros(n); src = np.empty(m); rho = np.empty(m)
    for v in range(n):
        p = psi[v]; fpsi[v] = -(p*p*p - p) + h - fcq[v]
    for e in range(m):
        a = ta[e]; b = hb[e]; zz = z[e]; A2 = zz.real*zz.real + zz.imag*zz.imag; Ad = np.sqrt(A2 + d2)
        dp = psi[b] - psi[a]; we = Ad/wp; gr = kappa*we*dp
        fpsi[a] += gr; fpsi[b] -= gr
        sf0 = -(1.0 + A2)*zz                                   # −g z − μ|z|² z (g = μ = 1)
        dzf = sf0 + beta[e]*lw[e] + br[e] - 0.5*kappa*dp*dp/wp*zz/Ad
        c = dzf - sf0
        pe = (psi[a] + psi[b])*0.5
        sf = sf0 + lc*pe*zz/(2.0*wp*wp)
        q = lc*(A2 - m0)/(4.0*wp*wp); fpsi[a] += q; fpsi[b] += q
        d = cs*sf + cc[e]*c; dz[e] = d
        src[e] = 2.0*(zz.real*d.real + zz.imag*d.imag)
        rho[e] = A2 if A2 > 0.05 else 0.05
    return dz, fpsi, src, rho

def install():
    import local60 as LC
    def F(s, z, psi, parts=False):
        if parts: return LC.T.Sys.F(s, z, psi, parts)
        mod = s.mod; f = s.fab
        if not hasattr(s, "_fast"):
            if not hasattr(mod, "_E2"):
                mod._E2 = np.ascontiguousarray(np.array(mod.E)); mod._S2 = np.ascontiguousarray(np.array(mod.S)); mod._QV = np.ascontiguousarray(mod.QV)
            assert not mod.pin.any()
            L = f.L; nt = f.nt
            # Kk для rfftn: та сама матриця, що в fabgen.fft_K, лише половина спектра за останньою віссю
            s._fast = dict(ta=np.ascontiguousarray(f.tail), hb=np.ascontiguousarray(f.head),
                           beta=np.ascontiguousarray(np.broadcast_to(np.asarray(mod.beta, float), (f.m,))))
            s._Kk = getattr(s.K, "Kk_half", None)
        cc = np.ascontiguousarray(np.broadcast_to(np.asarray(s.cc, np.complex128), (f.m,)))
        chi = ((1 + psi)/2)**2; dchi = (1 + psi)/2
        lw, br, fcq = qf2.qforce2(np.ascontiguousarray(z), mod._E2, mod._S2, mod._QV, chi, dchi, float(mod.eps0), mod.d2, mod.a, f.m, f.n)
        F_ = s._fast
        dz, fpsi, src, rho = _pre(z, psi, F_["ta"], F_["hb"], lw, br, fcq, F_["beta"], cc, complex(s.cs), float(mod.kappa), float(s.wp),
                                  float(mod.d2), float(mod.h), float(s.lc), float(s.m0), f.n)
        ks = s.K(src) if s._Kk is None else _krfft(s, src)
        ks = ks + (rho.sum() - s.S0)/(s.tc*len(rho))
        return dz - ks/(2*rho)*z, fpsi
    LC.Loc.F = F
    # rfft-версія K: та сама побудова, що fabgen.fft_K (скопійовано дослівно), плюс збереження Kk для _krfft
    import fabgen as G
    def fft_K(f, P, n):
        L = f.L; nn = f.n; nt = f.nt; W = np.zeros((nt, nt, L, L, L)); Pc = P.tocsr()
        for t in range(nt):
            row = Pc.getrow(t*nn)
            for c, w in zip(row.indices, row.data):
                t2, y = divmod(int(c), nn); W[t, t2][np.unravel_index(y, (L, L, L))] += w
        Pk = np.conj(np.fft.fftn(W, axes=(2, 3, 4))).transpose(2, 3, 4, 0, 1); Kk = np.linalg.matrix_power(Pk, n)
        def K(src):
            S = np.fft.fftn(src.reshape(nt, L, L, L), axes=(1, 2, 3)).transpose(1, 2, 3, 0)
            return np.real(np.fft.ifftn(np.einsum("...ab,...b->...a", Kk, S).transpose(3, 0, 1, 2), axes=(1, 2, 3))).reshape(-1)
        K.Kk_half = np.ascontiguousarray(Kk[:, :, :L//2 + 1]); return K
    G.fft_K = fft_K

def _krfft(s, src):
    f = s.fab; L = f.L; nt = f.nt
    S = np.fft.rfftn(src.reshape(nt, L, L, L), axes=(1, 2, 3)).transpose(1, 2, 3, 0)
    out = np.einsum("...ab,...b->...a", s._Kk, S).transpose(3, 0, 1, 2)
    return np.fft.irfftn(out, s=(L, L, L), axes=(1, 2, 3)).reshape(-1)
