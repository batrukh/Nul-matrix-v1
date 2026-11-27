"""(384) аналіз знімків: світ, мертві нитки на межі, перколяція (обгортка тора), місце прориву."""
import sys, glob, json; sys.argv = ['x']; sys.path[:0] = ['/mnt/project-files/model/s52/r351', '/mnt/project-files/model/s52']
import numpy as np, rq351f as Q
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
s, f, mod, z0, p0 = Q.run(L=16, check=1); N, L = 4096, 16; T, H = f.tail, f.head
D = (f.ijk[H] - f.ijk[T] + L//2) % L - L//2
def comps(mask_l, nodes):
    g = coo_matrix((np.ones(mask_l.sum()), (T[mask_l], H[mask_l])), shape=(N, N)); nc, lab = connected_components(g, directed=False)
    sz = np.bincount(lab[nodes], minlength=nc); return lab, sz
def wraps(inb, wl):
    adj = [[] for _ in range(N)]
    for e in np.where(wl)[0]: adj[T[e]].append((H[e], D[e])); adj[H[e]].append((T[e], -D[e]))
    start = np.where(inb)[0][0]; U = {start: np.zeros(3, int)}; st = [start]; w = np.zeros(3, bool)
    while st:
        a = st.pop()
        for b, d in adj[a]:
            u = U[a] + d
            if b not in U: U[b] = u; st.append(b)
            else: w |= (U[b] != u)
    return w.astype(int).tolist()
out = []; prev = None
for fn in sorted(glob.glob('/mnt/project-files/model/s52/r384/run/snap_*.npz')):
    c = np.load(fn, allow_pickle=True); t = float(c['t']); z = c['z']; psi = c['psi']; A = np.abs(z)**2
    nw = psi > 0; wl = nw[T] & nw[H]
    lab, sz = comps(wl, nw)
    if sz.max() == 0: out.append(dict(t=t, big=0)); continue
    bl = sz.argmax(); inb = nw & (lab == bl); big = int(inb.sum())
    edge = (~wl) & (inb[T] | inb[H])                      # нитки Матриці на межі найбільшого світу
    far = (~wl) & ~(inb[T] | inb[H])
    r = dict(t=t, big=big, nedge=int(edge.sum()), dead_edge=float((A[edge] < .01).mean()), Aedge=float(A[edge].mean()),
             dead_far=float((A[far] < .01).mean()) if far.any() else None, Afar=float(A[far].mean()) if far.any() else None,
             wrap=wraps(inb, wl & inb[T] & inb[H]))
    if prev is not None:                                    # нові вузли світу: скільки окремих «пальців»
        new = inb & ~prev; nl = new[T] & new[H]
        if new.sum():
            _, s2 = comps(nl, new); s2 = np.sort(s2[s2 > 0])[::-1]; r.update(nnew=int(new.sum()), new_top=s2[:3].tolist())
    prev = inb; out.append(r); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
json.dump(out, open('/mnt/project-files/model/s52/r384/ana/ana384.json', 'w'))
