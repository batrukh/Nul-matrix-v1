"""disp124: № 64 — закон дисперсії калібрувального «фотона» (91) в однорідному світі: ω(k) від |k| і напрямку θ.
(а) лінійна теорія: малий кут θ_μ(n) у площині «фотона» (O(2)), енергія ½·ka·Σθ̇² + kb·Σ_tri F², F — ротор на трикутнику;
    блохова матриця 3×3 (три ланки на вузол) → гілки: калібрувальна (ω = 0), «фотон» (ω ∝ k), оптична (шахова вгору/вниз).
(б) перевірка повним нелінійним кодом gauge91 (Φ★ + SO(5), той самий інтегратор): плоска хвиля мод «фотон», «глюон», «зламаний»,
    частота з часового ряду проєкції; поправка стрибка жаби ω_true = (2/dt)·sin(ω_num·dt/2)."""
import numpy as np, json, sys
import gauge91 as G
from phi90 import vacuum

EV = np.array([(1.0, 0.0), (0.5, np.sqrt(3)/2), (-0.5, np.sqrt(3)/2)])      # e0, e1, e2 = e1 − e0 у площині

def kcart(m1, m2, L):                                                      # k для θ ∝ exp(i·2π(m1·i + m2·j)/L)
    b1 = np.array([1.0, -1/np.sqrt(3)]); b2 = np.array([0.0, 2/np.sqrt(3)]); return 2*np.pi/L*(m1*b1 + m2*b2)

def bloch(k, ka=1.0, kb=1.0):
    p0, p1 = np.exp(1j*k @ EV[0]), np.exp(1j*k @ EV[1])
    C = np.array([[1, -1, p0], [-p1, p0, -p0]])                            # F_up = θ0 + θ2(a+e0) − θ1 ; F_dn = θ1(b) − θ0(c) − θ2(b)
    w2, v = np.linalg.eigh(2*kb/ka*(C.conj().T @ C))
    return np.sqrt(np.maximum(w2, 0)), v

def run_wave(m1, m2, L=48, kind="photon", ka=1.0, kb=1.0, aw=-1.0, amp=1e-3, T=0.0, dt=0.05, nper=1.5):
    P0 = np.diag(np.sort(np.linalg.eigvalsh(vacuum(aw)))); P = np.broadcast_to(P0, (L, L, 5, 5)).copy()
    a = np.full((L, L), aw); g = G.Gauge(a, np.ones((3, L, L)), np.ones((2, L, L)), ka, kb)
    pair = dict(photon=(3, 4), broken=(0, 3), gluon=(0, 1))[kind]; Gk = G.gen(*pair)
    k = kcart(m1, m2, L); _, v = bloch(k, ka, kb); pol = v[:, 1]                       # гілка «фотона» (середня; нижня — калібрувальна)
    i, j = np.meshgrid(np.arange(L), np.arange(L), indexing="ij"); ph = np.exp(2j*np.pi*(m1*i + m2*j)/L)
    mode = np.array([np.real(pol[m]*ph) for m in range(3)]); mode /= np.sqrt((mode**2).sum())
    U = np.array([G.cayley((amp*mode[m])[..., None, None]*Gk) for m in range(3)]); E = np.zeros((3, L, L, 5, 5)); Pi = np.zeros_like(P)
    if not T: T = nper*2*np.pi/bloch(k, ka, kb)[0][1] if kind != "broken" else 40.0
    nst = int(round(T/dt)); q = np.empty(nst + 1)
    th = lambda U: 0.5*(U[..., pair[0], pair[1]] - U[..., pair[1], pair[0]])
    q[0] = (th(U)*mode).sum()
    for s in range(nst):
        P, Pi, U, E = g.step(P, Pi, U, E, dt); q[s + 1] = (th(U)*mode).sum()
    t = np.arange(nst + 1)*dt; qq = (q - q.mean())*np.hanning(len(q))
    nf = 16*len(q); F = np.abs(np.fft.rfft(qq, nf)); w = 2*np.pi*np.fft.rfftfreq(nf, dt); w0 = w[np.argmax(F[1:]) + 1]
    from scipy.optimize import curve_fit                                   # уточнення: q(t) = A·cos(ωt) (старт зі спокою)
    (A, wn), _ = curve_fit(lambda t, A, w: A*np.cos(w*t), t, q, p0=(q[0], w0))
    wn = abs(wn); wt = 2/dt*np.sin(wn*dt/2)
    return dict(m=(m1, m2), k=float(np.hypot(*k)), th=float(np.degrees(np.arctan2(k[1], k[0]))), kind=kind,
                w=float(wt), w_lin=float(bloch(k, ka, kb)[0][1]), keep=float(np.abs(q).max()/abs(q[0]) if q[0] else 0))

if __name__ == "__main__":
    out = dict(lin=[], num=[])
    # (а) лінійна: |k| від 0,02 до 3,6, 13 напрямків від 0 до 60°
    ka, kb = 1.0, 1.0
    for kk in np.r_[np.linspace(0.02, 0.4, 20), np.linspace(0.5, 3.6, 32)]:
        for th in np.linspace(0, np.pi/3, 13):
            k = kk*np.array([np.cos(th), np.sin(th)]); w, _ = bloch(k, ka, kb)
            out["lin"].append(dict(k=float(kk), th=float(np.degrees(th)), w=[float(x) for x in w]))
    # (б) повний код: L = 24 (аргумент), кілька k у трьох напрямках
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    for m in [(1, 0), (2, 0), (3, 0), (4, 0), (6, 0), (8, 0), (1, 1), (2, 2), (3, 3), (4, 4), (2, 1), (4, 2)]:
        for kind in (("photon", "gluon", "broken") if m in [(1, 0), (4, 0), (2, 2)] else ("photon",)):
            r = run_wave(*m, L=L, kind=kind); out["num"].append(r); print(r, flush=True)
    json.dump(out, open(f"disp124_L{L}.json", "w"))
