"""(343) Чи тримає дисклінація непарний квант директора стріл B. Ґратка: nsec секторів трикутної ґратки по 60° (5 — конус, 6 — пласка, 7 — сідло), радіус R (кільце a+b ≤ R),
краї — Діріхле. Ефективне поле директора (як (313)): H = Σ_ребра (χ_u − η_e χ_v)² + m²Σχ² − V·Σ_{r≤rc} χ², η = −1 на ребрах, що перетинають розріз між сектором nsec−1 і 0, якщо tw=1
(B — «спін 3»: перехід через сектор множить на −1, тож навколо дисклінації скрут = (−1)^nsec; tw=1 лише для непарних nsec має сенс як потік). Зв'язаний стан: E0 < m² і не залежить від R."""
import json, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
def lattice(nsec, R):
    idx = {(0, 0, 0): 0}
    def canon(s, a, b):
        s %= nsec
        if a == 0 and b == 0: return (0, 0, 0)
        if b < 0: return canon(s - 1, 1, a - 1) if b == -1 else None
        if a == 0: return canon(s + 1, b, 0)
        return (s, a, b)
    for s in range(nsec):
        for a in range(1, R + 1):
            for b in range(0, R + 1 - a): idx[(s, a, b)] = len(idx)
    E = set(); OFF = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    for (s, a, b), i in idx.items():
        for da, db in OFF:
            c = canon(s, a + da, b + db) if (s, a, b) != (0, 0, 0) else canon(da % nsec if False else 0, *(0, 0))
            if (s, a, b) == (0, 0, 0): continue
            if c is None or c not in idx: continue
            j = idx[c]
            if i < j:
                cross = (c[0] != s) and not (c == (0, 0, 0))
                tw = cross and ((s == nsec - 1 and c[0] == 0) or (s == 0 and c[0] == nsec - 1))
                E.add((i, j, tw))
    # центр: сусіди — (s,1,0) для всіх s
    for s in range(nsec): E.add((0, idx[(s, 1, 0)], False))
    ring = np.zeros(len(idx), int)
    for (s, a, b), i in idx.items(): ring[i] = a + b
    return idx, list(E), ring
def spectrum(nsec, R, tw, m2=0.1, V=0.0, rc=1, k=4):
    idx, E, ring = lattice(nsec, R); n = len(idx)
    r, c, v = [], [], []; diag = np.full(n, m2) - V*(ring <= rc)
    for i, j, t in E:
        eta = -1.0 if (t and tw) else 1.0
        diag[i] += 1; diag[j] += 1; r += [i, j]; c += [j, i]; v += [-eta, -eta]
    deg = np.bincount(np.array([e[0] for e in E] + [e[1] for e in E]), minlength=n)
    H = sp.csr_matrix((v, (r, c)), shape=(n, n)) + sp.diags(diag)
    bnd = ring == R; keep = ~bnd; Hk = H[keep][:, keep]                 # Діріхле
    w, U = sla.eigsh(Hk, k=k, sigma=m2 - 3, which='LM')
    o = np.argsort(w); w, U = w[o], U[:, o]
    rk = ring[keep]; core = float((U[:, 0]**2)[rk <= 4].sum())
    return dict(E=[float(x) for x in w], core=core, n=int(n), deg_center=int(deg[0]))
if __name__ == "__main__":
    res = {}
    for nsec in (5, 6, 7):
        for tw in ((0, 1) if nsec % 2 else (0,)):
            for R in (15, 25, 35):
                q = spectrum(nsec, R, tw); key = f'n{nsec}_tw{tw}_R{R}'; res[key] = q
                print(key, 'E0', round(q['E'][0], 5), 'E0-m2 x R^2', round((q['E'][0] - 0.1)*R*R, 3), 'core', round(q['core'], 3), 'deg0', q['deg_center'], flush=True)
    json.dump(res, open('cone343.json', 'w'), indent=1)
