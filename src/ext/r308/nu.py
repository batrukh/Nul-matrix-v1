"""(309) кутовий момент зв'язаних станів (307): гармоніки e^{iνφ} непарних глюонних компонент c = c02 + i·c12 у рамці з розрізом."""
import numpy as np, json, sys
sys.path.insert(0, '/tmp/claude-0/-home-user/11ca481a-2b4a-53b5-a097-dab19d222dd0/scratchpad/r306'); sys.path.insert(0, '/mnt/project-files/model/s52/r268/code'); import frame
D_ = '../r306/R20'; m = np.load(f'{D_}/meta.npz'); d = np.load(sys.argv[1]); js = json.load(open(sys.argv[2]))['modes']
mass, NP, aidx, T, rw = m['mass'], int(m['NP']), m['aidx'], m['T'], m['rw']; WI, WJ = rw.shape
F, Zr, Pref, ref = frame.frames(m)
pe, pv = np.linalg.eigh(Pref); b3 = pv[:, :3]; ze, zv = np.linalg.eigh(b3.T @ Zr @ b3); b3 = b3 @ zv; eo, e0, e1 = b3[:, 0], b3[:, 1], b3[:, 2]
g = lambda a, b: (np.outer(a, b) - np.outer(b, a))/np.sqrt(2); G02, G12 = g(e0, eo), g(e1, eo)
vx, vy = float(m['cx']) - (m['ii'][0] + 0.5*m['jj'][0]), float(m['cy']) - np.sqrt(3)/2*m['jj'][0]
NU = np.arange(-7, 8) + 0.5; out = []
for k, q in enumerate(js):
    if not (q['odd'] > 0.5 and q['q2'] > 0.5 and q['w'] < 0.3): continue
    x = d['V'][:, k]/np.sqrt(mass); C, PHI, R = [], [], []
    for mm in range(3):
        for (a, b) in np.argwhere(aidx[mm] >= 0):
            Fm = F.get((a, b))
            if Fm is None: continue
            A = np.einsum('k,kij->ij', x[14*NP + 10*aidx[mm, a, b] + np.arange(10)], T); Ap = Fm.T @ A @ Fm
            X, Y = a + 0.5*b - vx, np.sqrt(3)/2*b - vy
            C.append(np.sum(Ap*G02) + 1j*np.sum(Ap*G12)); PHI.append(np.arctan2(Y, X)); R.append(rw[a, b])
    C, PHI, R = map(np.array, (C, PHI, R)); sel = (R > 4) & (R < 12)
    res = {}
    for lab, NUs in [('півцілі', np.arange(-4, 4) + 0.5), ('цілі', np.arange(-4, 5).astype(float))]:
        rb = np.digitize(R[sel], [6, 8, 10]); cols = []; names = []
        for bb in range(4):
            for nu in NUs: cols.append((rb == bb)*np.exp(1j*nu*PHI[sel])); names.append(nu)
        Am = np.array(cols).T; coef, *_ = np.linalg.lstsq(Am, C[sel], rcond=None)
        resid = np.linalg.norm(Am @ coef - C[sel])**2/np.linalg.norm(C[sel])**2
        pw = np.zeros(len(NUs))
        for i, nu in enumerate(names): pw[list(NUs).index(nu)] += abs(coef[i])**2*np.sum(abs(cols[i])**2)
        res[lab] = (resid, NUs, pw/pw.sum())
    r_h, NUs, pw = res['півцілі']; top = np.argsort(pw)[::-1][:3]
    print(f"ω {q['w']:.4f} залишок: півцілі {r_h:.3f}, цілі {res['цілі'][0]:.3f};", ' '.join(f"ν={NUs[t]:+.1f}:{pw[t]:.2f}" for t in top))
    out.append(dict(w=q['w'], resid_half=r_h, resid_int=res['цілі'][0], nu=NUs[top].tolist(), p=pw[top].tolist())); continue
    out.append(dict(w=q['w'], nu=NU[top].tolist(), p=pw[top].tolist()))
json.dump(out, open(sys.argv[3], 'w'))
