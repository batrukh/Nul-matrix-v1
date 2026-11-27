# sim34d.py — sim34 з регуляризацією |P| біля нуля
import numpy as np, sys, json
from scipy.integrate import solve_ivp
import simNlib as N
g, mu = 1.0, 1.0
import os
DELTA = float(os.environ.get("DELTA", "1e-6"))

class M34:
    """Модель (10), mode 'tri', але ψ передається лише цілими нитками: κ/2·Σ (|z_e|/w₊)(ψ_a − ψ_b)²."""
    def __init__(self, fab, beta=1.5, eps0=3.0, h=0.2, kappa=1.0):
        self.f = fab; self.beta, self.eps0, self.h, self.kappa = beta, eps0, h, kappa
        self.wp = N.wplus(beta); self.pin = np.zeros(fab.m, bool)
        self.d2 = DELTA**2                       # регуляризація |P| → √(|P|² + δ²) біля P = 0 (лише числова)
    def unpack(self, y):
        m = self.f.m; return y[:m] + 1j*y[m:2*m], y[2*m:]
    def pack(self, z, psi): return np.concatenate([z.real, z.imag, psi])
    def rhs(self, t, y):
        f = self.f; z, psi = self.unpack(y); a, b = f.tail, f.head
        A = np.abs(z); Ad = np.sqrt(A**2 + self.d2); chi = ((1+psi)/2)**2; dchi = (1+psi)/2
        we = Ad/self.wp; dpsi = psi[b] - psi[a]
        lap = np.bincount(a, we*dpsi, f.n) - np.bincount(b, we*dpsi, f.n)
        force = self.kappa*lap - (psi**3 - psi) + self.h
        p, q, r = f.TV.T
        Et = self.eps0*(chi[p] + chi[q] + chi[r])/3
        l1, l2, l3 = f.tri_links(z); P = l1*l2*l3; aP = np.sqrt(np.abs(P)**2 + self.d2); u1 = P/aP - 1
        brk = np.zeros(f.m, complex)
        for E_, S_, c in ((f.E1, f.S1, np.conj(l2*l3)*u1), (f.E2, f.S2, np.conj(l3*l1)*u1), (f.E3, f.S3, np.conj(l1*l2)*u1)):
            v = -Et*np.where(S_ > 0, c, np.conj(c))
            brk += np.bincount(E_, v.real, f.m) + 1j*np.bincount(E_, v.imag, f.m)
        cost = (aP - P.real)/3*self.eps0
        for nd in (p, q, r): force -= np.bincount(nd, dchi[nd]*cost, f.n)
        ph = z/Ad
        dz = -g*z - mu*A**2*z + self.beta*f.ww(z) + brk - 0.5*self.kappa*dpsi**2/self.wp*ph
        dz[self.pin] = 0.0                       # нитки розрізу тримаються порожніми, поки діє діра
        return np.concatenate([dz.real, dz.imag, force])
    def energy(self, y):
        f = self.f; z, psi = self.unpack(y); a, b = f.tail, f.head; A = np.abs(z)
        chi = ((1+psi)/2)**2; l1, l2, l3 = f.tri_links(z); P = l1*l2*l3; p, q, r = f.TV.T
        brkE = np.sum(self.eps0*(chi[p] + chi[q] + chi[r])/3*(np.sqrt(np.abs(P)**2 + self.d2) - P.real))
        return float(np.sum(g*A**2/2 + mu*A**4/4) + brkE - self.beta*np.sum(P.real)
                     + np.sum((psi**2-1)**2/4 - self.h*psi) + self.kappa/2*np.sum(np.sqrt(A**2 + self.d2)/self.wp*(psi[a]-psi[b])**2))

def run(mod, y0, T, S, rtol=1e-8):
    return solve_ivp(mod.rhs, (0, T), y0, method="BDF", rtol=rtol, atol=1e-10, t_eval=[T], jac_sparsity=S).y[:, -1]

def diag(mod, y, island=None):
    f = mod.f; z, psi = mod.unpack(y); hol = f.holonomy(z)
    l1, l2, l3 = f.tri_links(z)
    ok = (np.abs(l1) > 0.3*mod.wp) & (np.abs(l2) > 0.3*mod.wp) & (np.abs(l3) > 0.3*mod.wp)
    inb = psi > 0; tin = inb[f.TV[:, 0]] & inb[f.TV[:, 1]] & inb[f.TV[:, 2]]
    out = dict(frac=float(inb.mean()), k_all=float(hol[ok].sum()/(2*np.pi)), n_ok=int(ok.sum()),
               flux_bubble=float(hol[ok & tin].sum()/(2*np.pi)), wil=tuple(np.round(f.wilson(z), 3)))
    if island is not None:
        tis = island[f.TV[:, 0]] & island[f.TV[:, 1]] & island[f.TV[:, 2]]
        out["flux_island"] = float(hol[ok & tis].sum()/(2*np.pi))
        tout = ~(island[f.TV[:, 0]] | island[f.TV[:, 1]] | island[f.TV[:, 2]])
        out["flux_matrix"] = float(hol[ok & tout].sum()/(2*np.pi))
    return out

def experiment(L, k, eps0, R, Wd, T1, tear=True, seed=0):
    fab = N.Fabric(L); S = N.sparsity(fab)
    mod = M34(fab, eps0=0.0, h=0.0)
    rng = np.random.default_rng(100 + k + 7*seed)
    y = run(mod, mod.pack(N.matrix_state(fab, 1.5, k, rng), -np.ones(fab.n)), 100, S)   # Матриця з потоком k
    rn = N.disk(fab, 0); island = rn < R
    re = 0.5*(rn[fab.tail] + rn[fab.head])
    if not tear: ring = np.zeros(fab.m, bool)
    elif Wd <= 1: ring = island[fab.tail] != island[fab.head]        # мінімальний розріз
    else: ring = (re >= R) & (re < R + Wd)
    res = dict(L=L, k=k, eps0=eps0, R=R, tear=tear, n_island=int(island.sum()), n_ring=int(ring.sum()))
    res["matrix"] = diag(mod, y, island)
    z, psi = mod.unpack(y)
    psi = np.where(island, 1.0, -1.0) if tear else -np.tanh((rn - R)/np.sqrt(2))
    z = z.copy(); z[ring] = 0.0
    mod = M34(fab, eps0=eps0, h=0.2); mod.pin = ring.copy(); y = mod.pack(z, psi)
    if tear:
        y = run(mod, y, T1, S); res["isolated"] = diag(mod, y, island)       # ізоляція протягом T1
        z, psi = mod.unpack(y); z = z.copy()
        z[ring] = 0.1*mod.wp*np.exp(1j*rng.uniform(-np.pi, np.pi, ring.sum()))  # заростання: нитки народжуються
        mod.pin[:] = False; y = mod.pack(z, psi)
    T = 0.0
    while True:
        y = run(mod, y, 200, S); T += 200
        d = diag(mod, y, island); r_ = float(np.abs(mod.rhs(0, y)).max())
        if (d["frac"] in (0.0, 1.0) and r_ < 1e-6) or r_ < 1e-8 or T >= 4000: break
    res["final"] = d; res["res"] = r_; res["T"] = T
    return res

if __name__ == "__main__":
    L, k, eps0, R, Wd, T1 = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
    tear = sys.argv[7] == "1"; seed = int(sys.argv[8]) if len(sys.argv) > 8 else 0
    print(json.dumps(experiment(L, k, eps0, R, Wd, T1, tear, seed), ensure_ascii=False))
