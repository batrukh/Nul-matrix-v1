"""(306) власні моди оператора малих коливань: ω², локалізація в ядрі (r<4), частка Z-непарних компонент, частка фотона.
python3 modes.py DIR NEV ALPHA R0  (ALPHA — додаткова яма −α·exp(−(r/R0)²) для всіх ступенів свободи)"""
import numpy as np, sys, json, scipy.sparse as sp
from scipy.sparse.linalg import eigsh
sys.path.insert(0, '/mnt/project-files/model/s52/r268/code'); import gauge91 as G
D_, NEV = sys.argv[1], int(sys.argv[2]); ALPHA = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0; R0 = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
K = sp.load_npz(f'{D_}/K.npz'); B = sp.load_npz(f'{D_}/B.npz'); m = np.load(f'{D_}/meta.npz')
mass, sites, NP, NA, aidx, rw, P0, U0, S, T = (m[k] for k in ['mass', 'sites', 'NP', 'NA', 'aidx', 'rw', 'P0', 'U0', 'S', 'T']); NP, NA = int(NP), int(NA)
WI, WJ = int(m['WI']), int(m['WJ']); N = K.shape[0]
Mi = sp.diags(1/np.sqrt(mass)); C = sp.diags(np.sqrt(mass)) @ B
XI = 4.0; Dm = Mi @ K @ Mi + XI*(C @ C.T)
rr = np.zeros(N); rr[:14*NP] = np.repeat(rw[sites[:, 0], sites[:, 1]], 14); base = np.zeros((N, 2), int); base[:14*NP] = np.repeat(sites, 14, 0)
for k in range(3):
    for (a, b) in np.argwhere(aidx[k] >= 0): e = 14*NP + 10*aidx[k, a, b] + np.arange(10); rr[e] = rw[a, b]; base[e] = (a, b)
if ALPHA: Dm = Dm - sp.diags(ALPHA*np.exp(-(rr/R0)**2))
w2, V = eigsh(Dm.tocsc(), k=NEV, sigma=-0.02, which='LM'); o = np.argsort(w2); w2, V = w2[o], V[:, o]
sys.path.insert(0, '/tmp/claude-0/-home-user/11ca481a-2b4a-53b5-a097-dab19d222dd0/scratchpad/r306'); import frame
F, Zr, Pref, ref = frame.frames(m)
pe, pv = np.linalg.eigh(Pref); u, v = pv[:, 3], pv[:, 4]; Gph = (np.outer(u, v) - np.outer(v, u))/np.sqrt(2)
np.savez_compressed(f'{D_}/vec_a{ALPHA}_r{R0}.npz', w2=w2, V=V.astype(np.float32))
res = []
x = V/np.sqrt(mass)[:, None]
for k in range(NEV):
    tot = odd = ph = core = fr = 0.0
    xp = x[:14*NP, k].reshape(NP, 14); xa = x[14*NP:, k].reshape(NA, 10)
    dP = np.einsum('nk,kij->nij', xp, S)
    for s_i, (a, b) in enumerate(sites):
        Fm = F.get((a, b)); w = np.sum(dP[s_i]**2); tot += w; core += w*(rw[a, b] < 4)
        if Fm is not None: fr += w; Dp = Fm.T @ dP[s_i] @ Fm; odd += np.sum((0.5*(Dp - Zr @ Dp @ Zr))**2)
    for mm in range(3):
        for (a, b) in np.argwhere(aidx[mm] >= 0):
            A = np.einsum('k,kij->ij', xa[aidx[mm, a, b]], T); w = 0.5*np.sum(A**2); tot += w; core += w*(rw[a, b] < 4)
            Fm = F.get((a, b))
            if Fm is not None: fr += w; Ap = Fm.T @ A @ Fm; odd += 0.5*np.sum((0.5*(Ap - Zr @ Ap @ Zr))**2); ph += 0.5*np.sum(Ap*Gph)**2
    res.append(dict(w2=float(w2[k]), w=float(np.sign(w2[k])*np.sqrt(abs(w2[k]))), core=core/tot, odd=odd/max(fr, 1e-30), photon=ph/max(fr, 1e-30), framed=fr/tot))
json.dump(dict(alpha=ALPHA, R0=R0, modes=res, Zr=Zr.round(3).tolist()), open(f'{D_}/modes_a{ALPHA}_r{R0}.json', 'w'))
print('Zr', np.round(np.diag(Zr), 2))
for q in res:
    print(f"ω {q['w']:.4f} core {q['core']:.2f} odd {q['odd']:.2f} ph {q['photon']:.2f}")
