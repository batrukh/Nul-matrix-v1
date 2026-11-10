"""рамка паралельного перенесення в області (дерево BFS, що не перетинає промінь ліворуч від ядра) і голономія через розріз"""
import numpy as np, sys
sys.path.insert(0, '/mnt/project-files/model/s52/r268/code'); import gauge91 as G
def frames(m, verbose=False, rcore=3.5):
    sites, rw, U0, P0 = m['sites'], m['rw'], m['U0'], m['P0']; WI, WJ = rw.shape
    cxw = np.unravel_index(np.argmin(rw), rw.shape)
    X_ = np.arange(WI)[:, None] + 0.5*np.arange(WJ)[None, :]; Y_ = np.sqrt(3)/2*np.arange(WJ)[None, :] + 0*X_
    vx, vy = float(m['cx']) - (m['ii'][0] + 0.5*m['jj'][0]) , float(m['cy']) - np.sqrt(3)/2*m['jj'][0]
    inreg = np.zeros((WI, WJ), bool); inreg[sites[:, 0], sites[:, 1]] = True; inreg &= rw >= rcore          # ядро без рамки
    def crosses(p, q):
        (x1, y1), (x2, y2) = p, q
        if (y1 - vy)*(y2 - vy) >= 0: return False
        return x1 + (vy - y1)*(x2 - x1)/(y2 - y1) < vx
    ref = (cxw[0] + 8, cxw[1]); F = {ref: np.eye(5)}; queue = [ref]; cut = []
    while queue:
        n = queue.pop(0)
        for mm, d in enumerate(G.DIRS):
            for sgn in (1, -1):
                q = (n[0] + sgn*d[0], n[1] + sgn*d[1])
                if not (0 <= q[0] < WI and 0 <= q[1] < WJ) or not inreg[q]: continue
                if crosses((X_[n], Y_[n]), (X_[q], Y_[q])): cut.append((n, q, mm, sgn)); continue
                if q in F: continue
                F[q] = (U0[mm][n].T @ F[n]) if sgn == 1 else (U0[mm][q] @ F[n]); queue.append(q)
    hol = []
    for n, q, mm, sgn in cut:
        if n in F and q in F and rw[n] > 8:
            Tq = (U0[mm][n].T @ F[n]) if sgn == 1 else (U0[mm][q] @ F[n]); hol.append(F[q].T @ Tq)
    if verbose: print('vortex', vx, vy, 'centre site', cxw, 'cut', len(cut), 'hol', len(hol), [round(np.trace(h), 3) for h in hol[:8]])
    Zr = np.mean(hol, 0); ev, vv = np.linalg.eigh(0.5*(Zr + Zr.T)); Zr = vv @ np.diag(np.sign(ev)) @ vv.T
    Pref = F[ref].T @ P0[ref] @ F[ref]
    return F, Zr, Pref, ref
if __name__ == '__main__':
    m = np.load(sys.argv[1] + '/meta.npz'); F, Zr, Pref, ref = frames(m, True); print(np.round(np.linalg.eigvalsh(Zr), 3), len(F))
    print(np.round(Zr, 2)); print(np.round(Pref, 3))
