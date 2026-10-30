"""mat180: (180) спектр калібрувальної моди в Матриці (Φ = 0, a = +1, ваги ниток w = 0,02 — підлога варіанту «поле лише де протополе»).
(а) лінійна теорія: ваги w множать і кінетичну (ka·w), і магнітну (kb·w) частину → ω² = 2kb/ka·C†C, від w не залежить;
    Φ = 0 не ламає SO(5) → усі 10 генераторів безмасові, та сама дисперсія, що в «фотона» світу (124).
(б) перевірка повним нелінійним кодом gauge91 (той самий інтегратор): плоска хвиля для трьох генераторів, ваги 1 і 0,02, мала й велика амплітуда."""
import numpy as np, json, sys
import gauge91 as G
from disp124 import kcart, bloch

def run_wave(m1, m2, L=24, kind="photon", w=0.02, am=1.0, amp=1e-3, dt=0.05, nper=1.5, ka=1.0, kb=1.0):
    P = np.zeros((L, L, 5, 5)); a = np.full((L, L), am)
    g = G.Gauge(a, np.full((3, L, L), w), np.full((2, L, L), w), ka, kb)
    pair = dict(photon=(3, 4), broken=(0, 3), gluon=(0, 1))[kind]; Gk = G.gen(*pair)
    k = kcart(m1, m2, L); wl, v = bloch(k, ka, kb); pol = v[:, 1]
    i, j = np.meshgrid(np.arange(L), np.arange(L), indexing="ij"); ph = np.exp(2j*np.pi*(m1*i + m2*j)/L)
    mode = np.array([np.real(pol[m]*ph) for m in range(3)]); mode /= np.sqrt((mode**2).sum())
    amp_ = amp*np.sqrt(3*L*L)                                               # amp — типовий кут на ланці
    U = np.array([G.cayley((amp_*mode[m])[..., None, None]*Gk) for m in range(3)]); E = np.zeros((3, L, L, 5, 5)); Pi = np.zeros_like(P)
    T = nper*2*np.pi/wl[1]; nst = int(round(T/dt)); q = np.empty(nst + 1)
    th = lambda U: 0.5*(U[..., pair[0], pair[1]] - U[..., pair[1], pair[0]])
    q[0] = (th(U)*mode).sum(); leak = 0.0
    for s in range(nst):
        P, Pi, U, E = g.step(P, Pi, U, E, dt); q[s + 1] = (th(U)*mode).sum()
        leak = max(leak, float(np.abs(P).max()))
    t = np.arange(nst + 1)*dt; qq = (q - q.mean())*np.hanning(len(q))
    nf = 16*len(q); F = np.abs(np.fft.rfft(qq, nf)); ww = 2*np.pi*np.fft.rfftfreq(nf, dt); w0 = ww[np.argmax(F[1:]) + 1]
    from scipy.optimize import curve_fit
    (A, wn), _ = curve_fit(lambda t, A, w: A*np.cos(w*t), t, q, p0=(q[0], w0))
    wn = abs(wn); wt = 2/dt*np.sin(wn*dt/2)
    return dict(m=(m1, m2), k=float(np.hypot(*k)), kind=kind, w_wt=w, amp=amp, w=float(wt), w_lin=float(wl[1]),
                rel=float(wt/wl[1] - 1), phi_max=leak)

if __name__ == "__main__":
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 24; out = []
    for m in [(1, 0), (2, 0), (4, 0), (8, 0), (2, 2), (4, 2)]:
        for kind in ("photon", "gluon", "broken"):
            for w in (1.0, 0.02):
                r = run_wave(*m, L=L, kind=kind, w=w); out.append(r); print(r, flush=True)
    for amp in (0.05, 0.2, 0.5):                                            # нелінійність (неабелева самодія)
        for kind in ("photon", "broken"):
            r = run_wave(1, 0, L=L, kind=kind, w=0.02, amp=amp); out.append(r); print(r, flush=True)
    json.dump(out, open(f"mat180_L{L}.json", "w"))
