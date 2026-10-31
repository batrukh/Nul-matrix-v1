"""ff155: inertial mass of a vortex from free fall in an EXTERNAL strain gradient while the vortex sits in its own well.
N = 1 + s_own + s_ext, s_ext = g·(Ly/2π)·sin(2π(j − j0)/L) (dN/dy = g at the cores, as in ff152); s_own — from the source (153), K.
Force on the vortex = −(energy)·∇N ≈ −m·g; acceleration a = g·m/m_I → m_I/m = g/a. 20 units with friction (with the whole field), then 30 units without friction.
K = 0 — without its own well (as (152))."""
import sys, os, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import grav153 as GS, phi90 as PH
K, gg = float(sys.argv[1]), float(sys.argv[2]); T = 30.0
z = np.load("../r153/vort149_L96.npz"); P, U = z["P"].copy(), z["U"].copy(); L = 96; X, Y = PH.coords(L); Z = np.zeros_like
g0 = GS.GaugeMS(L, 1.0, 0, couple=False); eV = g0.edens(np.broadcast_to(GS.P0, P.shape), Z(P), np.broadcast_to(np.eye(5), U.shape), Z(U))
ex = g0.edens(P, Z(P), U, Z(U)) - eV; k0 = np.unravel_index(np.argmax(np.where(X < X.mean(), ex, -1)), ex.shape)
w = np.clip(ex, 0, None)*(PH.tdist(X, Y, X[k0], Y[k0], L) < 4); j0 = float((w*Y).sum()/w.sum())/(np.sqrt(3)/2)
jj = np.arange(L)[None, :]*np.ones((L, 1)); sext = gg*L*np.sqrt(3)/2/(2*np.pi)*np.sin(2*np.pi*(jj - j0)/L)
class GE(GS.GaugeMS):
    def update_s(s, P, Pi, U, E, frozen=None):
        if s.couple and frozen is None:
            r = s.rho(P, Pi, U, E); r = r - r.mean(); sself = -np.real(np.fft.ifft2(np.fft.fft2(r)*s.Gk))/s.K
        else: sself = np.zeros((L, L))
        s.sn = sself + sext; s.set_N(1.0 + s.sn)
g = GE(L, K if K > 0 else 1.0, 0, couple=K > 0); Pi = Z(P); E = Z(U); t = 0.0; dt = 0.05
def cy():
    e = g0.edens(P, Z(P), U, Z(U)) - eV; out = []
    for side in (X < X.mean(), X >= X.mean()):
        q = np.where(side, e, -1e9); k = np.unravel_index(np.argmax(q), q.shape); ww = np.clip(e, 0, None)*(PH.tdist(X, Y, X[k], Y[k], L) < 4); out.append(float((ww*Y).sum()/ww.sum()))
    return 0.5*(out[0] + out[1])
while t < 20 - 1e-9: P, Pi, U, E = g.step(P, Pi, U, E, dt, gam=1.0); t += dt
y0 = cy(); rec = [(0.0, 0.0)]; t = 0.0
while t < T - 1e-9:
    P, Pi, U, E = g.step(P, Pi, U, E, dt, gam=0.0); t += dt
    if abs(2*t - round(2*t)) < dt/2: rec.append((round(t, 2), cy() - y0))
r = np.array(rec); c = np.polyfit(r[:, 0], r[:, 1], 2); acc = 2*c[0]
json.dump(dict(K=K, g=gg, acc=acc, fit=c.tolist(), rec=rec, smin=float(g.sn.min())), open(f"ff155_K{K:g}_g{gg:g}.json", "w"))
print(K, gg, "acc", acc, "acc/g", acc/gg if gg else 0, "v0", c[1], "smin", g.sn.min(), flush=True)
