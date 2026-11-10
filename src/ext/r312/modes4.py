"""(312) моди оператора з директором: python3 modes4.py DIR NEV [vac]. Класифікація як у (307) для Φ і ланок; частка директора."""
import numpy as np, sys, json, scipy.sparse as sp
from scipy.sparse.linalg import eigsh
sys.path.insert(0, '/tmp/claude-0/-home-user/11ca481a-2b4a-53b5-a097-dab19d222dd0/scratchpad/r306'); import frame
D_, NEV = sys.argv[1], int(sys.argv[2]); VAC = len(sys.argv) > 3
K = sp.load_npz(f'{D_}/K.npz'); B = sp.load_npz(f'{D_}/B.npz'); m = np.load(f'{D_}/meta.npz')
mass, sites, NP, NA, ND, aidx, rw, S, T, n0 = (m[k] for k in ['mass', 'sites', 'NP', 'NA', 'ND', 'aidx', 'rw', 'S', 'T', 'n0']); NP, NA, ND = int(NP), int(NA), int(ND)
N = K.shape[0]; Mi = sp.diags(1/np.sqrt(mass)); C = sp.diags(np.sqrt(mass)) @ B
Dm = Mi @ K @ Mi + 4.0*(C @ C.T)
F, Zr, Pref, ref = frame.frames(m)
pe, pv = np.linalg.eigh(Pref); b3 = pv[:, :3]; u, v = pv[:, 3], pv[:, 4]
if VAC:
    nr = F[ref].T @ n0[ref]; eo_ = b3 @ (b3.T @ nr); eo_ /= np.linalg.norm(eo_); Zr = np.eye(5) - 2*np.outer(eo_, eo_) - 2*np.outer(u, u)
ze, zv = np.linalg.eigh(b3.T @ Zr @ b3); b3 = b3 @ zv; eo, e0, e1 = b3[:, 0], b3[:, 1], b3[:, 2]
gen = lambda a, b: (np.outer(a, b) - np.outer(b, a))/np.sqrt(2); Q = gen(e0, e1)*np.sqrt(2); Gph = gen(u, v)
prof = {}
for lo, hi in [(0, 3.5), (3.5, 6), (6, 10), (10, 14), (14, 20)]:
    prof[f'{lo}-{hi}'] = float(np.mean([(F[k].T @ n0[k] @ eo)**2 for k in F if lo <= rw[k] < hi])) if any(lo <= rw[k] < hi for k in F) else None
print('|n·e2|² за r:', {k: (round(v_, 4) if v_ is not None else None) for k, v_ in prof.items()}, 'Φ у рамці ref:', np.round(np.linalg.eigvalsh(Pref), 4))
import os; SIG = float(os.environ.get('SIG', -0.02)); w2, V = eigsh(Dm.tocsc(), k=NEV, sigma=SIG, which='LM'); o = np.argsort(w2); w2, V = w2[o], V[:, o]
x = V/np.sqrt(mass)[:, None]; res = []
for k in range(NEV):
    tot = odd = ph = core = fr = qq = 0.0
    xp = x[:14*NP, k].reshape(NP, 14); xd = x[14*NP:14*NP + ND, k]; xa = x[14*NP + ND:, k].reshape(NA, 10); dP = np.einsum('nk,kij->nij', xp, S)
    wd = np.sum(xd**2)
    for s_i, (a, b) in enumerate(sites):
        Fm = F.get((a, b)); w = np.sum(dP[s_i]**2); tot += w; core += w*(rw[a, b] < 4)
        if Fm is not None:
            fr += w; Dp = Fm.T @ dP[s_i] @ Fm; X = 0.5*(Dp - Zr @ Dp @ Zr); odd += np.sum(X**2); qq += np.sum((Q @ X - X @ Q)**2)
    for mm in range(3):
        for (a, b) in np.argwhere(aidx[mm] >= 0):
            A = np.einsum('k,kij->ij', xa[aidx[mm, a, b]], T); w = 0.5*np.sum(A**2); tot += w; core += w*(rw[a, b] < 4)
            Fm = F.get((a, b))
            if Fm is not None:
                fr += w; Ap = Fm.T @ A @ Fm; X = 0.5*(Ap - Zr @ Ap @ Zr); odd += 0.5*np.sum(X**2); qq += 0.5*np.sum((Q @ X - X @ Q)**2); ph += np.sum(Ap*Gph)**2
    res.append(dict(w2=float(w2[k]), w=float(np.sign(w2[k])*np.sqrt(abs(w2[k]))), core=core/tot, odd=odd/max(fr, 1e-30), photon=ph/max(fr, 1e-30), q2=qq/max(odd, 1e-30), dirw=wd/(wd + tot)))
json.dump(dict(prof=prof, modes=res), open(f'{D_}/m4.json', 'w')); np.savez_compressed(f'{D_}/m4vec.npz', w2=w2, V=V.astype(np.float32))
for q in res: print(f"ω {q['w']:.4f} core {q['core']:.2f} odd {q['odd']:.2f} ph {q['photon']:.2f} q2 {q['q2']:.2f} dir {q['dirw']:.2f}")
