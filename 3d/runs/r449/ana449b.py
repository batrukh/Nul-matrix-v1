"""(449) рішальний: перекриття острова з еталоном r425 M_seed{s} + нуль за 2000 періодичними зсувами; крутизна τ."""
import glob, json, numpy as np, sys
RD = sys.argv[1] if len(sys.argv) > 1 else 'runs'
sys.argv = ['x']; exec(open('ana449.py').read().split('rng = np.random')[0])
rng = np.random.default_rng(0); out = []
def island(psi):
    B = psi > 0
    if not (50 <= B.sum() <= 0.4*n): return None, ('world' if B.mean() > 0.4 else 'none')
    m = np.zeros(n, bool); m[comp(B)[0]] = True; return m, 'island'
for d in sorted(glob.glob(RD + '/s*_ns*')):
    fs = glob.glob(d + '/t3_*[0-9].npz')
    if not glob.glob(d + '/done') or not fs: continue
    s = int(d.split('/s')[1].split('_')[0]); ns = int(d.split('_ns')[1].split('_')[0])
    m, kind = island(np.load(fs[0])['psi']); r = dict(seed=s, ns=ns, kind=kind)
    if m is not None:
        ref, _ = island(np.load(glob.glob(f'/mnt/project-files/model/s52/r425/runs/M_seed{s}/t3_*.npz')[0])['psi'])
        ov = (m & ref).sum()/min(m.sum(), ref.sum()); M = m.reshape(L, L, L)
        null = np.array([(np.roll(M, tuple(rng.integers(0, L, 3)), (0, 1, 2)).ravel() & ref).sum()/min(m.sum(), ref.sum()) for _ in range(2000)])
        g = feats(G.tau_cubic(L, 4.0, s))['grad'].reshape(L, L, L)
        gn = np.array([g[np.roll(M, tuple(rng.integers(0, L, 3)), (0, 1, 2))].mean() for _ in range(2000)])
        r.update(size=int(m.sum()), overlap=round(float(ov), 3), null95=round(float(np.quantile(null, .95)), 3), same=bool(ov >= .5 and ov > np.quantile(null, .95)), grad_pct=round(float((gn < g[M].mean()).mean()), 3))
    out.append(r); print(r)
json.dump(out, open('ana449b.json' if RD == 'runs' else 'ana449b_' + RD + '.json', 'w'), indent=1)
