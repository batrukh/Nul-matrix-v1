"""(305) аналіз: голономія в локалізованому базисі (2 моди на кожному вихорі) і багаточастинкові фази Λ²U."""
import numpy as np, json, sys, itertools
out = {}
for tag in sys.argv[1:]:
    r = json.load(open(f'braid_{tag}.json')); U = np.array(r['U']); Q = np.array(r['Q'])
    Ul = Q.T @ U @ Q
    pairs = list(itertools.combinations(range(4), 2))
    W = np.array([[np.linalg.det(Ul[np.ix_(p, q)]) for q in pairs] for p in pairs])   # Λ²: <p|Λ²U|q>
    one = [k for k, p in enumerate(pairs) if p[0] < 2 <= p[1]]; same = [k for k, p in enumerate(pairs) if (p[1] < 2 or p[0] >= 2)]
    leak = float(np.abs(W[np.ix_(same, one)]).max())
    e_one = np.linalg.eigvals(W[np.ix_(one, one)]); e_same = np.linalg.eigvals(W[np.ix_(same, same)])
    blk = float(np.abs(Ul[:2, :2]).max() + np.abs(Ul[2:, 2:]).max())
    out[tag] = dict(phases1=r['phases_over_pi'], xq=np.round(r['xq'], 2).tolist(), diag_blocks=blk, Ul=np.round(Ul, 3).tolist(),
                    two_one_per_vortex=sorted((np.angle(e_one)/np.pi).round(3).tolist()), two_same=sorted((np.angle(e_same)/np.pi).round(3).tolist()),
                    leak=leak, smin=r['smin'], Emax=r['Emax'], gap=r['gap'])
    print(tag, json.dumps(out[tag], ensure_ascii=False))
json.dump(out, open('ana305.json', 'w'))
