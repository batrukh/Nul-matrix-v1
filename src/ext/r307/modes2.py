"""(307) маса непарним глюонам G02, G12 (фотон G34 без маси) у операторі (306).
python3 modes2.py DIR NEV MG [VAC]. Маса діє на ланки з рамкою (r>=3,5): у ядрі «поле-утримувач» зникає, як хіггс у ядрі вихору.
Класифікація мод: core, odd (Z), photon, q2 = |[Q,X]|²/|X|² для непарної частини (Q = G01: глюон 1, фотон 0)."""
import numpy as np, sys, json, scipy.sparse as sp
from scipy.sparse.linalg import eigsh
sys.path.insert(0, '/tmp/claude-0/-home-user/11ca481a-2b4a-53b5-a097-dab19d222dd0/scratchpad/r306'); import frame
D_, NEV, MG = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]); VAC = len(sys.argv) > 4 and sys.argv[4] == "vac"; import os; RC = float(os.environ.get("RC", 3.5))
K = sp.load_npz(f'{D_}/K.npz'); B = sp.load_npz(f'{D_}/B.npz'); m = np.load(f'{D_}/meta.npz')
mass, sites, NP, NA, aidx, rw, S, T = (m[k] for k in ['mass', 'sites', 'NP', 'NA', 'aidx', 'rw', 'S', 'T']); NP, NA = int(NP), int(NA)
N = K.shape[0]; Mi = sp.diags(1/np.sqrt(mass)); C = sp.diags(np.sqrt(mass)) @ B
Dm = Mi @ K @ Mi + 4.0*(C @ C.T)
F, Zr, Pref, ref = frame.frames(m, rcore=RC)
pe, pv = np.linalg.eigh(Pref); b3 = pv[:, :3]; u, v = pv[:, 3], pv[:, 4]
if VAC: Zr = np.eye(5); Zr = Zr - 2*np.outer(b3[:, 2], b3[:, 2]) - 2*np.outer(u, u)
z3 = b3.T @ Zr @ b3; ze, zv = np.linalg.eigh(z3); b3 = b3 @ zv          # перший — Z-непарний напрям трійки
eo, e0, e1 = b3[:, 0], b3[:, 1], b3[:, 2]
gen = lambda a, b: (np.outer(a, b) - np.outer(b, a))/np.sqrt(2)
Gl = [gen(e0, eo), gen(e1, eo)] + ([gen(e0, e1)] if os.environ.get("ALL3") else []); Q = gen(e0, e1)*np.sqrt(2); Gph = gen(u, v)
rows, cols, vals = [], [], []
for mm in range(3):
    for (a, b) in np.argwhere(aidx[mm] >= 0):
        Fm = F.get((a, b))
        if Fm is None: continue
        e = 14*NP + 10*aidx[mm, a, b]
        cs = np.array([[np.sum(T[k]*(Fm @ g @ Fm.T)) for k in range(10)] for g in Gl])
        q, _ = np.linalg.qr(cs.T); blk = MG**2*(q @ q.T)
        ii, jj = np.meshgrid(np.arange(10), np.arange(10), indexing='ij'); rows += list(e + ii.ravel()); cols += list(e + jj.ravel()); vals += list(blk.ravel())
Dm = Dm + sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
w2, V = eigsh(Dm.tocsc(), k=NEV, sigma=-0.02, which='LM'); o = np.argsort(w2); w2, V = w2[o], V[:, o]
x = V/np.sqrt(mass)[:, None]; res = []
for k in range(NEV):
    tot = odd = ph = core = fr = qq = 0.0
    xp = x[:14*NP, k].reshape(NP, 14); xa = x[14*NP:, k].reshape(NA, 10); dP = np.einsum('nk,kij->nij', xp, S)
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
    res.append(dict(w2=float(w2[k]), w=float(np.sign(w2[k])*np.sqrt(abs(w2[k]))), core=core/tot, odd=odd/max(fr, 1e-30), photon=ph/max(fr, 1e-30), q2=qq/max(odd, 1e-30)))
tag = ("vac" if VAC else "v") + (f"_rc{RC}" if RC != 3.5 else "") + ("_all3" if os.environ.get("ALL3") else "")
json.dump(dict(MG=MG, modes=res), open(f'{D_}/m2_{tag}_mg{MG}.json', 'w')); np.savez_compressed(f'{D_}/m2vec_{tag}_mg{MG}.npz', w2=w2, V=V.astype(np.float32))
for q in res: print(f"ω {q['w']:.4f} core {q['core']:.2f} odd {q['odd']:.2f} ph {q['photon']:.2f} q2 {q['q2']:.2f}")
