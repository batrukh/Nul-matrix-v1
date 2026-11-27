"""disp64: № 64 крок 2 — малі хвилі в однорідному спокійному світі (w64). Стан обертає фази ниток з частотою Ω_c (свій для кожного класу),
що є калібрувальним поворотом; лінеаризація в обертовій системі: M = J − iΩ_c. Для хвильового вектора k (m1, m2 в оберненій ґратці)
— блоховий базис (3 класи ниток × Re/Im × cos/sin + ψ × cos/sin = 14), M(k) 14×14 скінченними різницями, власні числа λ = −γ + iω."""
import numpy as np, sys
import w64
def setup(fn, L=48, s0=8.0, ac=1.0, tilt=1):
    s, _, _ = w64.build(L, s0, ac, tilt=tilt); d = np.load(fn); z0, p0 = d["z"], d["psi"]; f = s.fab; c = w64.cls(f, L)
    dz, _ = s.F(z0, p0); Om = np.array([(dz/(1j*z0))[c == k].real.mean() for k in range(3)])
    dd = f.xy[f.head] - f.xy[f.tail]; dd = (dd + L//2) % L - L//2; mid = f.xy[f.tail] + dd/2
    return s, z0, p0, c, Om, mid
def Mk(st, m1, m2, eps=1e-5):
    s, z0, p0, c, Om, mid = st; f = s.fab; L = f.L if hasattr(f, "L") else int(np.sqrt(f.n))
    phe = 2*np.pi*(m1*mid[:, 0] + m2*mid[:, 1])/L; phv = 2*np.pi*(m1*f.xy[:, 0] + m2*f.xy[:, 1])/L
    B = []
    for k in range(3):
        for u in (1.0, 1j):
            for tr in (np.cos, np.sin): B.append((u*tr(phe)*(c == k), np.zeros(f.n)))
    for tr in (np.cos, np.sin): B.append((np.zeros(f.m, complex), tr(phv)))
    nrm = [float((np.abs(b[0])**2).sum() + (b[1]**2).sum()) for b in B]; B = [b for b, q in zip(B, nrm) if q > 1e-9]; nrm = [q for q in nrm if q > 1e-9]
    Omz = Om[c]
    def J(b):
        a1 = s.F(z0 + eps*b[0], p0 + eps*b[1]); a2 = s.F(z0 - eps*b[0], p0 - eps*b[1])
        return ((a1[0] - a2[0])/(2*eps) - 1j*Omz*b[0], (a1[1] - a2[1])/(2*eps))
    M = np.zeros((len(B), len(B)))
    for i, b in enumerate(B):
        r = J(b)
        for j, bj in enumerate(B): M[j, i] = (np.real(np.conj(bj[0])*r[0]).sum() + (bj[1]*r[1]).sum())/nrm[j]
    return M
def kvec(m1, m2, L):
    b1 = np.array([1.0, -1/np.sqrt(3)]); b2 = np.array([0.0, 2/np.sqrt(3)]); return 2*np.pi/L*(m1*b1 + m2*b2)
if __name__ == "__main__":
    fn = sys.argv[1]; ac = float(sys.argv[2]); L = 48
    st = setup(fn, ac=ac); print("Ω класів", np.round(st[4], 4))
    M0 = Mk(st, 0, 0); print("k=0:", np.round(np.sort_complex(np.linalg.eigvals(M0)), 4))
    for sh in ([(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)], [(1, 1), (-1, 2), (-2, 1), (-1, -1), (1, -2), (2, -1)], [(2, 0), (0, 2), (-2, 2), (-2, 0), (0, -2), (2, -2)]):
        for m in sh:
            k = kvec(*m, L); ev = np.linalg.eigvals(Mk(st, *m)); ev = ev[np.argsort(-ev.real)]
            print("m", m, "|k| %.4f θ %6.1f" % (np.hypot(*k), np.degrees(np.arctan2(k[1], k[0]))), " ".join("%+.4f%+.4fi" % (e.real, e.imag) for e in ev[:6]), flush=True)
