"""imp181: (181) the domain|Matrix boundary as a weight jump: fraction of the energy of a "photon" wave that passes the boundary at normal incidence.
Strip Lx × Ly (torus): domain at x0 ≤ i < x1 (ψ = +1), Matrix outside (ψ = −1), boundary ψ = tanh(distance/δ); link weights max(χ, floor), a(ψ) as in gauge91.
The Φ background relaxes with friction (Tr), then a "photon" packet (O(2) of the domain) e^{−(i−ic)²/2σ²}·cos(k i), uniform in j → fronts parallel to the boundary.
Packet energy = difference of energy densities of the perturbed and unperturbed runs (as gauge91.run); fractions in the domain / Matrix versus time.
Comparison with the impedance estimate T = 4w/(1+w)² (step boundary, long wave)."""
import numpy as np, json, sys
import gauge91 as G
from phi90 import vacuum
from disp124 import bloch

def run(floor=0.02, kidx=0.6, Lx=256, Ly=4, x0=96, x1=160, delta=1.0, sig=8.0, aw=-1.0, am=1.0, ka=1.0, kb=1.0, dt=0.05, T=110.0, Tr=20.0, amp=1e-3):
    i = np.arange(Lx)[:, None]*np.ones((1, Ly)); d = np.minimum(i - x0 + 0.5, x1 - 0.5 - i)       # >0 inside the domain
    psi = np.tanh(d/delta); chi = 0.5*(1 + np.tanh(psi/0.2)); a = am + (aw - am)*chi
    wl, wt = G.weights(chi, floor); g = G.Gauge(a, wl, wt, ka, kb)
    P0 = np.diag(np.sort(np.linalg.eigvalsh(vacuum(aw)))); P = (chi[..., None, None] > 0.5)*P0[None, None]
    U = np.broadcast_to(np.eye(5), (3, Lx, Ly, 5, 5)).copy(); E = np.zeros((3, Lx, Ly, 5, 5)); Pi = np.zeros_like(P); t = 0.0
    while t < Tr - 1e-9: P, Pi, U, E = g.step(P, Pi, U, E, dt, gam=1.0); t += dt
    Pi[:] = 0; E[:] = 0
    k = np.array([kidx, -kidx/np.sqrt(3)]); w_, v = bloch(k, ka, kb); pol = v[:, 1]                 # k·e0 = kidx, k·e1 = 0
    ic = 0.5*(x0 + x1); env = np.exp(-(i - ic)**2/(2*sig**2)); ph = np.exp(1j*kidx*i)
    mode = np.array([np.real(pol[m]*ph)*env for m in range(3)]); mode *= amp/np.abs(mode).max()
    Gk = G.gen(3, 4)                                                                                  # "photon" in the Φ★ basis (diagonal, sorted)
    UB = np.array([G.cayley(mode[m][..., None, None]*Gk) for m in range(3)]) @ U
    A = [P.copy(), Pi.copy(), U.copy(), E.copy()]; B = [P.copy(), Pi.copy(), UB, E.copy()]
    inw = d > 0; rec = []; t = 0.0
    while t < T - 1e-9:
        A = list(g.step(*A, dt)); B = list(g.step(*B, dt)); t += dt
        if abs(t - round(t)) < dt/2:
            e = g.edens(*B) - g.edens(*A); tot = e.sum()
            rec.append(dict(t=round(t), tot=float(tot), fw=float(e[inw].sum()/tot), prof=[float(x) for x in e.sum(1)]))
    vg = None
    return dict(floor=floor, kidx=kidx, delta=delta, w=float(w_[1]), T_imp=4*floor/(1 + floor)**2, rec=rec)

if __name__ == "__main__":
    floor, kidx, delta = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    r = run(floor, kidx, delta=delta); fm = [1 - q["fw"] for q in r["rec"]]
    print(f"floor {floor} k {kidx} delta {delta}: imp {r['T_imp']:.4f}  fM(t=40,60,80,100) " +
          " ".join(f"{fm[t-1]:.4f}" for t in (40, 60, 80, 100)) + f"  drift {r['rec'][-1]['tot']/r['rec'][0]['tot']-1:.1e}", flush=True)
    json.dump(r, open(f"imp181_f{floor:g}_k{kidx:g}_d{delta:g}.json", "w"))
