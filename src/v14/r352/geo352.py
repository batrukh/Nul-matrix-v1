"""(352) ідея автора 12.11: довжина нитки з її сили, l = (w+²/|z|²)^p (сильна — коротша). Кривина 2D: дефіцит кута у вузлі
δ_v = 2π − Σ кутів трикутників (закон косинусів). Чи δ_v ∝ енергії у вузлі (E_v − Ē)? Знімки r348 (база 8.11 без Φ★, s0 3,5, L96).
На торі Σδ = 0 точно (Гаусс–Бонне), тому порівнюємо з E − Ē. Енергія вузла: нитки навпіл, трикутники по третині, ψ і градієнт."""
import sys, json, glob, numpy as np
sys.path.insert(0, '/mnt/project-files/model/s52/r348/code'); import simNlib as N, rnd51 as R
B, EPS0, H, KAP, g, mu = 1.3, 3.0, 0.3, 1.0, 1.0, 1.0; WP = N.wplus(B); W2 = WP**2
f0 = N.Fabric(96); L = 96; n = f0.n; p_, q_, r_ = f0.TV.T; f = f0
def node_energy(z, psi):
    A2 = np.abs(z)**2; a, b = f.tail, f.head
    el = g*A2/2 + mu*A2**2/4; E = np.bincount(a, el/2, n) + np.bincount(b, el/2, n)
    l1, l2, l3 = f.tri_links(z); P = l1*l2*l3; chi = ((1+psi)/2)**2
    et = -B*P.real + EPS0*(chi[p_]+chi[q_]+chi[r_])/3*(np.abs(P) - P.real)
    for v in (p_, q_, r_): E += np.bincount(v, et/3, n)
    E += (psi**2-1)**2/4 - H*psi
    eg = KAP/2*np.abs(z)/WP*(psi[a]-psi[b])**2; E += np.bincount(a, eg/2, n) + np.bincount(b, eg/2, n)
    return E
def deficit(z, p):
    l = (W2/np.maximum(np.abs(z)**2, 1e-6))**p
    a, b, c = l[f.E1], l[f.E2], l[f.E3]          # сторони трикутника (p,q), (q,r), (r,p)
    bad = (a >= b + c) | (b >= a + c) | (c >= a + b)
    ang = lambda x, y, o: np.arccos(np.clip((x*x + y*y - o*o)/(2*x*y), -1, 1))   # кут між сторонами x, y навпроти o
    D = np.full(n, 2*np.pi)
    D -= np.bincount(p_, ang(a, c, b), n); D -= np.bincount(q_, ang(a, b, c), n); D -= np.bincount(r_, ang(b, c, a), n)
    return D, int(bad.sum())
def lap(x):                                      # графовий лапласіан вузлового поля
    a, b = f.tail, f.head; d = x[b] - x[a]; return np.bincount(a, d, n) - np.bincount(b, d, n)
def block(x, k): return x.reshape(L//k, k, L//k, k).mean((1, 3)).ravel()
def corr(x, y): return float(np.corrcoef(x, y)[0, 1])
out = {}
for run in ['ge0.0025_L96_s1', 'ge0.0025_L96_s2', 'ge0_L96_s1', 'lat_ge0_L96_s1']:
    fn = glob.glob(f'/mnt/project-files/model/s52/r348/{run}/*_sp.npz')[0]; D_ = np.load(fn); ts = D_['t']
    bad_ = np.load(fn.replace('_sp.npz', '.npz'))['bad']; f = N.Fabric(96); R.flip(f, bad_)      # стріли прогону (перевернуті нитки)
    for p in (0.05, 0.1, 0.25):
        rec = dict(); acc = []
        for i in range(len(ts)-10, len(ts)):
            z, psi = D_['z'][i], D_['psi'][i]; E = node_energy(z, psi); dE = E - E.mean()
            dl, nb = deficit(z, p); A2n = (np.bincount(f.tail, np.abs(z)**2, n) + np.bincount(f.head, np.abs(z)**2, n))/6
            acc.append((dl, dE, A2n, psi, nb))
        dl = np.mean([x[0] for x in acc], 0); dE = np.mean([x[1] for x in acc], 0); A2n = np.mean([x[2] for x in acc], 0); psi = np.mean([x[3] for x in acc], 0)
        dl1, dE1 = acc[-1][0], acc[-1][1]
        W = psi > 0; bd = np.zeros(n, bool); bd[f.tail[W[f.tail] != W[f.head]]] = True; bd[f.head[W[f.tail] != W[f.head]]] = True
        rec.update(t=float(ts[-1]), bad_tri=int(acc[-1][4]), sum_def=float(dl1.sum()),
            r_inst=corr(dl1, dE1), r_avg=corr(dl, dE), r_lapA2=corr(dl, -lap(A2n)), r_lapE=corr(dl, -lap(dE)),
            **{f'r_blk{k}': corr(block(dl, k), block(dE, k)) for k in (4, 8, 16)},
            slope_blk8=float(np.polyfit(block(dE, 8), block(dl, 8), 1)[0]),
            mean_def=dict(world=float(dl[W & ~bd].mean()), border=float(dl[bd].mean()), matrix=float(dl[~W & ~bd].mean())),
            mean_dE=dict(world=float(dE[W & ~bd].mean()), border=float(dE[bd].mean()), matrix=float(dE[~W & ~bd].mean())),
            std_def=float(dl.std()), frac_world=float(W.mean()))
        out[f'{run}_p{p}'] = rec; print(run, p, json.dumps(rec), flush=True)
        if p == 0.1 and run == 'ge0.0025_L96_s1': np.savez('/mnt/project-files/model/s52/r352/maps_s1.npz', dl=dl, dE=dE, psi=psi, A2n=A2n)
json.dump(out, open('/mnt/project-files/model/s52/r352/ana352.json', 'w'), indent=1)
