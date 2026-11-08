"""(286) Струна Аліси у справжньому світі бази P0,16 (ℓ8, γ0,0025, b0,1): стан r276 g0.0025_s{SEED}_T600 при t590.
Вихор садимо калібрувально (як (285)): ланки, що перетинають відрізок від найглибшої точки світу поперек пучка до Матриці (Φ★≈0),
отримують Z = I − 2(v₂v₂ᵀ + v₃v₃ᵀ) з власних векторів Φ★ вузла (R(π) у площині «найбільший від'ємний – найменший додатний»).
Кінець розрізу — у Матриці, де вакууму нема, тож другого вихора у світі нема. ψ і нитки заморожені (як (282)). ctrl — те саме без розрізу.
Пучок — уздовж головної осі світу біля вихора (PCA), джерело за D вузлів перед вихором; рамка «фотона» в джерелі — перенесенням по ланках.
Вимір без глобальної рамки: C = Σ⟨δE_v, δE_c⟩ / Σ⟨δE_c, δE_c⟩ по ланках вузла і часу (фон обох копій однаковий поза розрізом).
python3 realw.py SEED LAM  (vort і ctrl в одному процесі)"""
import os as _o; _V = _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), 'variant'); RUNS = _o.environ.get('NM_RUNS', _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), '..', 'runs', 'out', 'variant'))
import sys, os, glob, pickle, json, time, numpy as np, scipy.ndimage as ndi
sys.path.insert(0, _V)
import gauge91 as G, phi90 as PH, matter98 as M98
seed, LAM = int(sys.argv[1]), float(sys.argv[2])
L = 96; dt = 0.05; TR = float(os.environ.get('TR', '40')); T = float(os.environ.get('T', '40')); D = float(os.environ.get('D', '12'))
AMP = 0.02; SPAR = LAM; SPERP = float(os.environ.get('SPERP', '5')); CPH = 0.65; EXC = 7.0
F = json.load(open(glob.glob(RUNS + f'/g0.0025_s{seed}_T600/*_fine.json')[0]))
old = [c for t_, cl in F if 585 <= t_ <= 590 for c in cl]
C = pickle.load(open(glob.glob(RUNS + f'/ck/g0.0025_s{seed}_T600/*_ck.pkl')[0], 'rb'))
GF = C['GF']; psi = C['psi'].reshape(L, L)
X, Y = PH.coords(L)
dist = ndi.distance_transform_edt(np.tile(psi > 0, (3, 3)))[L:2*L, L:2*L]
ok = dist.copy()
for (ox, oy) in old:
    r, _, _ = M98.wrapd(X, Y, ox, oy, L); ok[r < EXC] = -1
pv = np.unravel_index(np.argmax(ok), ok.shape); vx, vy = float(X[pv]), float(Y[pv])
# головна вісь світу біля вихора
r, dx, dy = M98.wrapd(X, Y, vx, vy, L); m_ = (r < 14) & (dist >= 2)
cov = np.cov(np.vstack([dx[m_], dy[m_]])); w_, v_ = np.linalg.eigh(cov); ax = v_[:, 1]; pe = np.array([-ax[1], ax[0]])
# розріз: від вихора вздовж +pe до Матриці (+4 вузли за межею)
s_end = 0.0
while True:
    s_end += 0.5; qx, qy = vx + s_end*pe[0], vy + s_end*pe[1]
    rr, _, _ = M98.wrapd(X, Y, qx, qy, L); k = np.unravel_index(np.argmin(rr), rr.shape)
    if psi[k] < -0.5 or s_end > 40: break
s_end += 4.0
I5 = np.eye(5)
U00, P00, Pi00, E00 = GF.U.copy(), GF.P.copy(), GF.Pi.copy(), GF.E.copy()
chi = 0.5*(1 + np.tanh(psi/0.2)); wl, wt = G.weights(chi, GF.floor)
g = G.Gauge(1.0 - 2.0*chi, wl, wt, GF.ka, GF.kb, GF.b)
def setup(mode):
    U = U00.copy(); ncut = 0
    if mode == 'vort':
        for m, ev in enumerate(EV):
            A = np.array([[ev[0], -pe[0]], [ev[1], -pe[1]]]); Ai = np.linalg.inv(A)
            u = Ai[0, 0]*(-dx) + Ai[0, 1]*(-dy); s = Ai[1, 0]*(-dx) + Ai[1, 1]*(-dy)   # n + u e = v + s pe, dx = X_n − v
            hit = (u > 0) & (u < 1) & (s > 0) & (s < s_end) & (r < s_end + 3)
            for (i, j) in np.argwhere(hit):
                w, vv = np.linalg.eigh(P00[i, j]); vv = vv[:, np.argsort(w)]
                Z = I5 - 2*(np.outer(vv[:, 2], vv[:, 2]) + np.outer(vv[:, 3], vv[:, 3]))
                U[m][i, j] = Z @ U[m][i, j]; ncut += 1
    P, Pi, E = P00.copy(), Pi00.copy(), E00.copy(); t = 0.0
    while t < TR - 1e-9: P, Pi, U, E = g.step(P, Pi, U, E, dt, gam=0.2); t += dt
    Pi[:] = 0; E[:] = 0
    return P, Pi, U, E, ncut
EV = [(1.0, 0.0), (0.5, np.sqrt(3)/2), (-0.5, np.sqrt(3)/2)]
SV = setup('vort'); SC = setup('ctrl'); ncut = SV[4]
e_rel = g.edens(*SV[:4]) - g.edens(*SC[:4])
DIRS = [(1, 0), (0, 1), (-1, 1)]
def walk(P, U, i, j, R):
    # шестикутник радіуса R навколо (i, j): старт у (i+R, j−R)... кроки ґратки; перенос рамки площини «фотона» по ланках
    def pl(Pn): w, vv = np.linalg.eigh(Pn); return vv[:, np.argsort(w)][:, 3:5]
    i, j = (i + R) % L, (j - R) % L; F0 = pl(P[i, j]); Fq = F0.copy(); mn = 1.0
    for (di, dj) in [(0, 1)]*R + [(-1, 1)]*R + [(-1, 0)]*R + [(0, -1)]*R + [(1, -1)]*R + [(1, 0)]*R:
        if (di, dj) in DIRS: m = DIRS.index((di, dj)); Fn = U[m][i, j].T @ Fq; i, j = (i + di) % L, (j + dj) % L
        else: m = DIRS.index((-di, -dj)); i, j = (i + di) % L, (j + dj) % L; Fn = U[m][i, j] @ Fq
        Pl = pl(P[i, j]); uu, ss, vt = np.linalg.svd(Pl.T @ Fn); mn = min(mn, ss.min()); Fq = Pl @ (uu @ vt)
    return float(np.linalg.det(F0.T @ Fq)), float(mn)
HOL = {f'{nm}_R{R}': walk(S[0], S[2], pv[0], pv[1], R) for nm, S in (('vort', SV), ('ctrl', SC)) for R in (2, 3, 4)}
print('голономія', HOL, flush=True)
P, Pi, U, E = SC[:4]          # рамку джерела беремо з контролю, ту саму — для обох
# джерело
sx, sy = vx - D*ax[0], vy - D*ax[1]
rs, dxs, dys = M98.wrapd(X, Y, sx, sy, L); ks = np.unravel_index(np.argmin(rs), rs.shape)
# рамка «фотона»: перенесення по ланках від центру джерела (BFS у межах 3σ)
def plane(Pn): w, vv = np.linalg.eigh(Pn); return vv[:, np.argsort(w)][:, 3:5]
Fr = np.zeros((L, L, 5, 2)); seen = np.zeros((L, L), bool); Fr[ks] = plane(P[ks]); seen[ks] = True; q = [ks]
reg = rs < 3*max(SPAR, SPERP) + 2
NB = [((1, 0), 0, +1), ((0, 1), 1, +1), ((-1, 1), 2, +1), ((-1, 0), 0, -1), ((0, -1), 1, -1), ((1, -1), 2, -1)]
while q:
    i, j = q.pop(0)
    for (di, dj), m, sg in NB:
        a, b = (i + di) % L, (j + dj) % L
        if seen[a, b] or not reg[a, b]: continue
        Fn = U[m][i, j].T @ Fr[i, j] if sg > 0 else U[m][a, b] @ Fr[i, j]   # U[m][n] переносить з n+e у n
        Pl = plane(P[a, b]); uu, ss, vt = np.linalg.svd(Pl.T @ Fn); Fr[a, b] = Pl @ (uu @ vt); seen[a, b] = True; q.append((a, b))
Gk = np.einsum('...i,...j->...ij', Fr[..., 0], Fr[..., 1]); Gk = Gk - np.swapaxes(Gk, -1, -2)
k = 2*np.pi/LAM
def theta(m, shift):
    ev = EV[m]; _, ddx, ddy = M98.wrapd(X + ev[0]/2, Y + ev[1]/2, sx + shift*ax[0], sy + shift*ax[1], L)
    par = ddx*ax[0] + ddy*ax[1]; per = ddx*pe[0] + ddy*pe[1]
    A_ = AMP*np.cos(k*par)*np.exp(-par**2/(2*SPAR**2))*np.exp(-per**2/(2*SPERP**2))*reg
    return A_*(pe[0]*ev[0] + pe[1]*ev[1])           # поперечна поляризація
def run(S):
    P, Pi, U, E = [x.copy() for x in S[:4]]
    UB, EB = U.copy(), E.copy(); h = 1e-3
    for m in range(3):
        th = theta(m, 0.0); thd = (theta(m, CPH*h) - th)/h
        UB[m] = G.cayley(th[..., None, None]*Gk) @ U[m]; EB[m] = thd[..., None, None]*Gk
    return [P, Pi, U, E, P.copy(), Pi.copy(), UB, EB]
RV, RC = run(SV), run(SC)
num = np.zeros((L, L)); den = np.zeros((L, L)); flv = np.zeros((L, L)); kin = []; t = 0.0; t0 = time.time()
while t < T - 1e-9:
    for R_ in (RV, RC):
        R_[0:4] = g.step(*R_[0:4], dt); R_[4:8] = g.step(*R_[4:8], dt)
    t += dt
    dv = RV[7] - RV[3]; dc = RC[7] - RC[3]
    num += dt*np.sum(dv*dc, axis=(0, -2, -1)); den += dt*np.sum(dc*dc, axis=(0, -2, -1)); flv += dt*np.sum(dv*dv, axis=(0, -2, -1))
    if abs(t - round(t)) < dt/2 and round(t) % 4 == 0: kin.append(np.stack([np.sum(dv*dv, axis=(0, -2, -1)), np.sum(dc*dc, axis=(0, -2, -1)), np.sum(dv*dc, axis=(0, -2, -1))]).astype(np.float32))
out = f'rw_s{seed}_lam{LAM:g}'
np.savez_compressed(out + '.npz', num=num, den=den, flv=flv, kin=np.array(kin), psi=psi.astype(np.float32), dist=dist.astype(np.float32), e_rel=e_rel.astype(np.float32), X=X, Y=Y)
json.dump(dict(seed=seed, lam=LAM, v=[vx, vy], depth=float(dist[pv]), ax=ax.tolist(), pe=pe.tolist(), s_end=s_end, src=[sx, sy],
               src_depth=float(dist[ks]), ncut=ncut, hol=HOL, e_core=float(e_rel[r < 4].sum()), e_cut=float(e_rel.sum()), old=old, TR=TR, T=T, D=D, sec=time.time() - t0), open(out + '.json', 'w'))
print(out, 'v', (round(vx, 1), round(vy, 1)), 'depth', dist[pv], 'src depth', dist[ks], 'ncut', ncut, 'cut len', s_end, 'sec', round(time.time() - t0), flush=True)
