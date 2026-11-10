"""(309) кутовий момент зв'язаних станів (307). На вузлі: векторне поле глюона V = (2/3)Σ_m c_m d̂_m, c = c02 + i·c12 (заряд G01),
компоненти u± = Vx ± i·Vy (спін ±1). Підгонка гармоніками e^{iνφ} (ν цілі / півцілі) у кільцях r 4–12; J = ν ± 1."""
import numpy as np, json, sys
sys.path.insert(0, '/tmp/claude-0/-home-user/11ca481a-2b4a-53b5-a097-dab19d222dd0/scratchpad/r306'); sys.path.insert(0, '/mnt/project-files/model/s52/r268/code'); import frame, gauge91 as GG
D_ = sys.argv[4] if len(sys.argv) > 4 else '../r306/R20'
m = np.load(f'{D_}/meta.npz'); d = np.load(sys.argv[1]); js = json.load(open(sys.argv[2]))['modes']
mass, NP, aidx, T, rw = m['mass'], int(m['NP']), m['aidx'], m['T'], m['rw']
F, Zr, Pref, ref = frame.frames(m)
pe, pv = np.linalg.eigh(Pref); b3 = pv[:, :3]; ze, zv = np.linalg.eigh(b3.T @ Zr @ b3); b3 = b3 @ zv; eo, e0, e1 = b3[:, 0], b3[:, 1], b3[:, 2]
g = lambda a, b: (np.outer(a, b) - np.outer(b, a))/np.sqrt(2); G02, G12 = g(e0, eo), g(e1, eo)
vx, vy = float(m['cx']) - (m['ii'][0] + 0.5*m['jj'][0]), float(m['cy']) - np.sqrt(3)/2*m['jj'][0]
dh = np.array([[d_[0] + 0.5*d_[1], np.sqrt(3)/2*d_[1]] for d_ in GG.DIRS])
out = []
def fit(Cv, PHI, R, NUs):
    rb = np.digitize(R, [6, 8, 10]); cols, names = [], []
    for bb in range(4):
        for nu in NUs: cols.append((rb == bb)*np.exp(1j*nu*PHI)); names.append(nu)
    Am = np.array(cols).T; coef, *_ = np.linalg.lstsq(Am, Cv, rcond=None)
    pw = np.zeros(len(NUs))
    for i, nu in enumerate(names): pw[list(NUs).index(nu)] += abs(coef[i])**2*np.sum(abs(cols[i])**2)
    return np.linalg.norm(Am @ coef - Cv)**2/np.linalg.norm(Cv)**2, pw
for k, q in enumerate(js):
    if not (q['odd'] > 0.5 and q['q2'] > 0.5 and q['w'] < float(sys.argv[5] if len(sys.argv) > 5 else 0.3)): continue
    x = d['V'][:, k]/np.sqrt(mass); up, um, PHI, R = [], [], [], []
    for (a, b) in np.argwhere((rw > 4) & (rw < 12)):
        Fm = F.get((a, b))
        if Fm is None or any(aidx[mm, a, b] < 0 for mm in range(3)): continue
        c = []
        for mm in range(3):
            A = np.einsum('k,kij->ij', x[14*NP + 10*aidx[mm, a, b] + np.arange(10)], T); Ap = Fm.T @ A @ Fm
            c.append(np.sum(Ap*G02) + 1j*np.sum(Ap*G12))
        V = (2/3)*np.einsum('m,mi->i', np.array(c), dh)
        up.append(V[0] + 1j*V[1]); um.append(V[0] - 1j*V[1])
        PHI.append(np.arctan2(np.sqrt(3)/2*b - vy, a + 0.5*b - vx)); R.append(rw[a, b])
    up, um, PHI, R = map(np.array, (up, um, PHI, R))
    line = f"ω {q['w']:.4f}"; rec = dict(w=q['w'])
    for lab, u, s in [('u+', up, +1), ('u-', um, -1)]:
        rh, pw = fit(u, PHI, R, np.arange(-4, 4) + 0.5); ri, _ = fit(u, PHI, R, np.arange(-4, 5).astype(float))
        NUs = np.arange(-4, 4) + 0.5; t = np.argsort(pw)[::-1][:2]; wt = np.sum(abs(u)**2)
        line += f" | {lab} вага {wt/(np.sum(abs(up)**2)+np.sum(abs(um)**2)):.2f} залишок ½:{rh:.3f} ціл:{ri:.3f} ν {NUs[t[0]]:+.1f}({pw[t[0]]:.2f}) {NUs[t[1]]:+.1f}({pw[t[1]]:.2f})"
        rec[lab] = dict(resid_half=rh, resid_int=ri, nu=NUs[t].tolist(), p=pw[t].tolist(), weight=float(wt))
    print(line); out.append(rec)
json.dump(out, open(sys.argv[3], 'w'))
