"""(345) Чи «хоче» енергія мати кривину поруч. Крапля світу ψ>0 (φ⁴, як ψ у моделі: V = ψ⁴/4 − ψ²/2, κ/2 Σ(ψa−ψb)², κ 1) у Матриці ψ<0 зі збереженою Σψ (сталий розмір)
на ґратці з дисклінацією (5 — конус, 7 — сідло, 6 — пласка). (1) Енергія краплі з центром на дисклінації проти далеко від неї. (2) Відпускаємо краплю на відстані d і дивимось, куди вона повзе.
Очікування з геометрії: на конусі коло довкола вершини коротше в √(5/6)·…(для тієї ж площі) — крапля дешевша, тож притягується; сідло — навпаки."""
import sys, json, numpy as np, scipy.sparse as sp
sys.path.insert(0, '/mnt/project-files/model/s52/r343'); from cone343 import lattice
def build(nsec, R):
    idx, E, ring = lattice(nsec, R); n = len(idx); P = np.zeros((n, 2))
    for (s, a, b), i in idx.items():
        if (s, a, b) == (0, 0, 0): continue
        e1 = np.array([np.cos(2*np.pi/nsec*s), np.sin(2*np.pi/nsec*s)]); e2 = np.array([np.cos(2*np.pi/nsec*(s+1)), np.sin(2*np.pi/nsec*(s+1))])
        P[i] = a*e1 + b*e2                        # вкладення лише для вибору стартової точки (відстані — графові)
    r_, c_ = [], []
    for i, j, t in E: r_ += [i, j]; c_ += [j, i]
    A = sp.csr_matrix((np.ones(len(r_)), (r_, c_)), shape=(n, n)); D = np.asarray(A.sum(1)).ravel()
    return A, D, ring, P, n
def bfsdist(A, src):
    import scipy.sparse.csgraph as cg
    return cg.shortest_path(A, unweighted=True, indices=src)
def energy(psi, A, D, kap=1.0):
    lap_e = 0.5*kap*(D*psi**2 - psi*(A @ psi)).sum()          # = κ/2 Σ_ребра (ψa−ψb)²
    return float(lap_e + (psi**4/4 - psi**2/2).sum())
def relax(psi, A, D, fix, steps=4000, dt=0.1, kap=1.0):
    for _ in range(steps):
        g = kap*(D*psi - A @ psi) + psi**3 - psi
        g[fix] = 0; g[~fix] -= g[~fix].mean()                      # збережена Σψ (поза закріпленим краєм)
        psi = psi - dt*g
    return psi
def centroid_dist(psi, dist0):
    w = np.clip(psi, 0, None); return float((w*dist0).sum()/w.sum())
if __name__ == "__main__":
    R = 70; r0 = 12; res = {}
    for nsec in (6, 5, 7):
        A, D, ring, P, n = build(nsec, R); fix = ring >= R - 1
        d0 = bfsdist(A, 0)                                         # графова відстань до центру (дисклінації)
        # точка далеко: вузол на кільці 34 у секторі 0
        far = [i for i in range(n) if ring[i] == 40][0]; dfar = bfsdist(A, far)
        out = {}
        K = 900; rng = np.random.default_rng(0); tb = 1e-3*rng.random(n)
        def ball(dd): sel = np.argsort(dd + tb)[:K]; p = -np.ones(n); p[sel] = 1.0; return p
        for tag, dd in (('centre', d0), ('far', dfar)):
            psi = ball(dd); psi[fix] = -1.0
            psi = relax(psi, A, D, fix); out[tag] = dict(E=energy(psi, A, D), area=int((psi > 0).sum()))
        # відпускаємо краплю зі зсувом 6 від центру
        start = [i for i in range(n) if ring[i] == 21][0]; ds = bfsdist(A, start)
        psi = ball(ds); psi[fix] = -1.0; traj = []
        for k in range(16):
            traj.append(float(d0[psi > 0].min())); psi = relax(psi, A, D, fix, steps=1500)
        traj.append(float(d0[psi > 0].min()))
        out['drift'] = traj; res[nsec] = out
        print(nsec, out['centre'], out['far'], 'dE(centre-far)', round(out['centre']['E'] - out['far']['E'], 4), 'drift', [round(x, 2) for x in traj], flush=True)
    json.dump(res, open('drop345.json', 'w'), indent=1)
