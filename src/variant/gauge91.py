"""gauge91: протополе Φ★ (як phi90) + калібрувальне поле SO(5) на нитках (класична гратчаста теорія Янга–Міллса–Хіггса).
Перевірка (не постулат), 22.10.2026: чи стають безмасовими калібрувальні хвилі незламаного сектору O(3) × O(2) («фотон» — O(2)).
    H = Σ_n [½|Π|² + V(Φ)] + Σ_l [(1/3)|UΦ'Uᵀ − Φ|² + (ka·w_l/4)|E|²] + Σ_tri kb·w_t·(5 − tr H_tri)
    Φ̇ = Π, U̇ = E·U (E антисиметрична), Π̇ = −∂H/∂Φ, Ė = −(2/(ka·w_l))·P_A[(∂H/∂U)·Uᵀ].
Ланка (n, μ) переносить Φ з n+μ у n; напрямки μ: (1,0), (0,1), (−1,1) — ті самі нитки, що в simNlib.Fabric.
Ваги w: 1 всюди (варіант «поле всюди») або χ(ψ) з підлогою (варіант «поле лише де протополе»).
Інтегратор: стрибок жаби (leapfrog), U оновлюється перетворенням Келі (точно ортогональне)."""
import numpy as np, json, sys
from phi90 import sym0, dV, vacuum, coords, tdist

DIRS = [(1, 0), (0, 1), (-1, 1)]
OWN = None                                                         # (145): маска світу, де стартує хвиля (необов'язково)
I5 = np.eye(5)
T_ = lambda A: np.swapaxes(A, -1, -2)
PA = lambda X: 0.5*(X - T_(X))
tr = lambda A: np.trace(A, axis1=-2, axis2=-1)
def sh(A, d): return np.roll(A, (-d[0], -d[1]), axis=(0, 1))     # значення у вузлі n+d
def put(A, d): return np.roll(A, (d[0], d[1]), axis=(0, 1))       # віднести значення, обчислене в n, до вузла n+d

def cayley(X):
    return np.linalg.solve(I5 - X/2, I5 + X/2)

class Gauge:
    def __init__(s, a, wl, wt, ka=1.0, kb=1.0, b=0.0):
        s.b = b; s.a = a; s.wl = wl; s.wt = wt; s.ka = ka; s.kb = kb      # wl (3,L,L), wt (2,L,L): вгору, вниз (індекс — базовий вузол a)

    def hol(s, U):
        U0, U1, U2 = U
        Hu = U0 @ sh(U2, DIRS[0]) @ T_(U1)                          # a → b=a+e0 → c=a+e1 → a
        Hd = sh(U1, DIRS[0]) @ T_(sh(U0, DIRS[1])) @ T_(sh(U2, DIRS[0]))   # b → d=b+e1 → c → b
        return Hu, Hd

    def edens(s, P, Pi, U, E):
        e = 0.5*np.sum(Pi**2, axis=(-2, -1))
        P2 = P @ P; t2 = tr(P2)
        e += s.a/2*t2 - s.b/3*tr(P2 @ P) + t2**2/4 + 0.5/4*tr(P2 @ P2)   # (123): кубічний член b
        for m, d in enumerate(DIRS):
            D = U[m] @ sh(P, d) @ T_(U[m]) - P
            e += (1/3)*np.sum(D**2, axis=(-2, -1)) + s.ka*s.wl[m]/4*np.sum(E[m]**2, axis=(-2, -1))
        Hu, Hd = s.hol(U)
        e += s.kb*(s.wt[0]*(5 - tr(Hu)) + s.wt[1]*(5 - tr(Hd)))
        return e

    def forces(s, P, U):
        FP = -dV(P, s.a, s.b)
        R = [np.zeros_like(U[0]) for _ in range(3)]                # R_l = (∂H/∂U_l)·U_lᵀ
        for m, d in enumerate(DIRS):
            X = U[m] @ sh(P, d) @ T_(U[m]); D = X - P
            FP += (2/3)*D
            FP -= put((2/3)*(T_(U[m]) @ D @ U[m]), d)
            R[m] += (4/3)*(D @ X)
        U0, U1, U2 = U
        bu = (s.kb*s.wt[0])[..., None, None]; bd = (s.kb*s.wt[1])[..., None, None]
        A, B = U0, sh(U2, DIRS[0]); Hu = A @ B @ T_(U1)            # вгору: H = A·B·Cᵀ, A = U0(a), B = U2(b), C = U1(a)
        R[0] += -bu*T_(Hu)
        R[2] += put(-bu*(T_(A) @ T_(Hu) @ A), DIRS[0])
        R[1] += -bu*Hu
        A, B, C = sh(U1, DIRS[0]), sh(U0, DIRS[1]), sh(U2, DIRS[0]); Hd = A @ T_(B) @ T_(C)   # вниз: H = A·Bᵀ·Cᵀ, A = U1(b), B = U0(c), C = U2(b)
        R[1] += put(-bd*T_(Hd), DIRS[0])
        R[0] += put(-bd*(T_(C) @ Hd @ C), DIRS[1])
        R[2] += put(-bd*Hd, DIRS[0])
        FE = np.array([-(2/(s.ka*s.wl[m]))[..., None, None]*PA(R[m]) for m in range(3)])
        return sym0(FP), FE

    def step(s, P, Pi, U, E, dt, gam=0.0):
        FP, FE = s.forces(P, U)
        Pi = Pi + 0.5*dt*(FP - gam*Pi); E = E + 0.5*dt*(FE - gam*E)
        P = sym0(P + dt*Pi); U = cayley(dt*E) @ U
        FP, FE = s.forces(P, U)
        Pi = (Pi + 0.5*dt*FP)/(1 + 0.5*dt*gam); E = (E + 0.5*dt*FE)/(1 + 0.5*dt*gam)
        return P, Pi, U, E

def weights(chi, floor):
    w = np.maximum(chi, floor)
    wl = np.array([0.5*(w + sh(w, d)) for d in DIRS])
    wt = np.array([(w + sh(w, DIRS[0]) + sh(w, DIRS[1]))/3, (sh(w, DIRS[0]) + sh(w, (1, 1)) + sh(w, DIRS[1]))/3])
    return wl, wt

def gen(i, j):
    G = np.zeros((5, 5)); G[i, j], G[j, i] = 1, -1; return G

def run(src="", L=48, aw=-1.0, am=1.0, mode="world", var="all", floor=0.02, kind="photon", T=30.0, dt=0.05, amp=0.05, r0=2.0,
        ka=1.0, kb=1.0, cen="", Tr=40.0, seed=1):
    if src:
        d0 = np.load(src); psi = d0["psi"]; P = d0["P"].copy(); L = psi.shape[0]
    else:
        psi = np.full((L, L), 1.0 if mode == "world" else -1.0)
        P = np.broadcast_to(np.diag(np.sort(np.linalg.eigvalsh(vacuum(aw)))) if mode == "world" else np.zeros((5, 5)), (L, L, 5, 5)).copy()
    chi = 0.5*(1 + np.tanh(psi/0.2)); a = am + (aw - am)*chi
    wl, wt = weights(chi, floor) if var == "world" else (np.ones((3, L, L)), np.ones((2, L, L)))
    g = Gauge(a, wl, wt, ka, kb)
    U = np.broadcast_to(I5, (3, L, L, 5, 5)).copy(); E = np.zeros((3, L, L, 5, 5)); Pi = np.zeros_like(P)
    t = 0.0
    if src:                                                        # спільна релаксація Φ і калібрувального поля (тертя)
        while t < Tr - 1e-9: P, Pi, U, E = g.step(P, Pi, U, E, dt, gam=1.0); t += dt
        Pi[:] = 0; E[:] = 0
    X, Y = coords(L)
    if cen: cx, cy = [float(v) for v in cen.split(",")]
    else: cx, cy = X.mean(), Y.mean()
    r = tdist(X, Y, cx, cy, L); k0 = np.unravel_index(np.argmin(r), r.shape)
    w_, v_ = np.linalg.eigh(P[k0]); v_ = v_[:, np.argsort(w_)]
    pair = dict(photon=(3, 4), broken=(0, 3), gluon=(0, 1))[kind]
    Gk = np.outer(v_[:, pair[0]], v_[:, pair[1]]); Gk = Gk - Gk.T
    # гладкий вихровий векторний потенціал A = amp·exp(−r²/2r0²)·(−Δy, Δx)/r0 (магнітний потік у плямі), θ_μ = A(середина ланки)·e_μ;
    # поштовх лише однієї ланки збуджував би головно «оптичну» (шахову між трикутниками вгору/вниз) гілку
    UB = U.copy(); EV = [(1.0, 0.0), (0.5, np.sqrt(3)/2), (-0.5, np.sqrt(3)/2)]
    for m, ev in enumerate(EV):
        mx, my = X + ev[0]/2, Y + ev[1]/2; best = None
        for p in (-1, 0, 1):
            for q in (-1, 0, 1):
                dx_, dy_ = mx - cx + p*L + q*L/2, my - cy + q*np.sqrt(3)/2*L; rr = np.hypot(dx_, dy_)
                if best is None: best = (rr, dx_, dy_)
                else:
                    k_ = rr < best[0]; best = (np.where(k_, rr, best[0]), np.where(k_, dx_, best[1]), np.where(k_, dy_, best[2]))
        rr, dx_, dy_ = best; Am = amp*np.exp(-rr**2/(2*r0**2))/r0
        th = Am*(-dy_*ev[0] + dx_*ev[1])
        UB[m] = cayley(th[..., None, None]*Gk) @ U[m]
    PA_, PiA, UA, EA = P.copy(), Pi.copy(), U.copy(), E.copy(); PB, PiB, EB = P.copy(), Pi.copy(), E.copy()
    bins = np.arange(0, L/2 + 1, 1.0); cnt = np.maximum(np.histogram(r, bins)[0], 1); rec = []; t = 0.0
    inw = psi > 0; e0A = g.edens(PA_, PiA, UA, EA).sum(); e0B = g.edens(PB, PiB, UB, EB).sum()
    while t < T - 1e-9:
        PA_, PiA, UA, EA = g.step(PA_, PiA, UA, EA, dt); PB, PiB, UB, EB = g.step(PB, PiB, UB, EB, dt); t += dt
        if abs(t*2 - round(t*2)) < dt/2:
            e = np.abs(g.edens(PB, PiB, UB, EB) - g.edens(PA_, PiA, UA, EA))
            prof = np.histogram(r, bins, weights=e)[0]/cnt; hot = e > 1e-3*e.max()
            rec.append(dict(t=round(t, 2), prof=[float(v) for v in prof], tot=float(e.sum()), fin=float(e[inw].sum()/e.sum()),
                            reach_w=float(r[hot & inw].max()) if (hot & inw).any() else 0.0,
                            reach_m=float(r[hot & ~inw].max()) if (hot & ~inw).any() else 0.0))
            if OWN is not None: rec[-1]["fown"] = float(e[OWN].sum()/e.sum())     # (145): частка у власному світі (без сусідніх світів)
    dE = float((g.edens(PB, PiB, UB, EB).sum() - e0B)/max(abs(e0B - e0A), 1e-30))
    name = f"gauge91_{('src_' + src.split('/')[-1][6:-4]) if src else 'uni' + mode}_{var}_{kind}_ka{ka:g}_kb{kb:g}" + (f"_f{floor:g}" if var == "world" else "")
    json.dump(dict(rec=rec, drift=dE), open(name + ".json", "w"))
    return name, rec, dE

if __name__ == "__main__":
    kw = dict(x.split("=") for x in sys.argv[1:])
    kw = {k: (v if k in ("src", "kind", "cen", "mode", "var") else (int(v) if k in ("L", "seed") else float(v))) for k, v in kw.items()}
    name, rec, dE = run(**kw); print(name, "drift/kick-energy %.2e" % dE)
    for q in rec[1::6]:
        p = np.array(q["prof"]); print(q["t"], "peak", int(np.argmax(p[2:]) + 2), "front", int(np.nonzero(p > 1e-3*p.max())[0].max()),
                                       "tot %.3g fin %.2f rw %.1f rm %.1f" % (q["tot"], q["fin"], q["reach_w"], q["reach_m"]), flush=True)
