"""(308)/(311) пробне поле-«директор» n (одиничний 5-вектор, n ≡ −n) на тлі вихору бази r294 (Φ, U заморожені).
E_n = Σ_ланки −(k/2)(nₓᵀ U n_{x+d})² + κ Σ nᵀΦn.  κ>0 тягне n у трійку Φ; вісь у трійці вибирає вихор.
python3 director.py DIR ORIENT(e2|e0) K KAP  [env ETA: −η·nᵀ(Σ(DΦ)²)n — директор відчуває напругу Φ у ядрі]  → DIR/dir_{ORIENT}_k{K}.npz (n, E)"""
import numpy as np, sys
sys.path.insert(0, '/mnt/project-files/model/s52/r268/code'); import gauge91 as G
sys.path.insert(0, '/tmp/claude-0/-home-user/11ca481a-2b4a-53b5-a097-dab19d222dd0/scratchpad/r306'); import frame
D_, OR, K, KAP = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]); VAC = D_.split('/')[-1].startswith('V')
m = np.load(f'{D_}/meta.npz'); P, U = m['P0'], m['U0']; WI, WJ = P.shape[:2]
if VAC: F = {(i, j): np.eye(5) for i in range(WI) for j in range(WJ)}; Zr = np.diag([1., 1, -1, -1, 1]); Pref = P[0, 0]
else: F, Zr, Pref, ref = frame.frames(m, rcore=3.5)
pe, pv = np.linalg.eigh(Pref); b3 = pv[:, :3]; ze, zv = np.linalg.eigh(b3.T @ Zr @ b3); b3 = b3 @ zv; eo, e0 = b3[:, 0], b3[:, 1]
v = eo if OR == 'e2' else e0
n = np.zeros((WI, WJ, 5)); rng = np.random.default_rng(1)
for i in range(WI):
    for j in range(WJ):
        Fm = F.get((i, j))
        if Fm is not None: n[i, j] = Fm @ v
        else: w, Vv = np.linalg.eigh(P[i, j]); n[i, j] = Vv[:, :3] @ rng.normal(size=3)
n /= np.linalg.norm(n, axis=-1, keepdims=True)
import os
ETA = float(os.environ.get('ETA', 0))
Mst = np.zeros_like(P)                       # «напруга» Φ: Σ_d (DΦ)², DΦ = U Φ(n+d) Uᵀ − Φ(n) (обидва напрямки)
for mm, d in enumerate(G.DIRS):
    a = slice(max(0, -d[0]), WI - max(0, d[0])), slice(max(0, -d[1]), WJ - max(0, d[1]))
    b = slice(max(0, d[0]), WI + min(0, d[0])), slice(max(0, d[1]), WJ + min(0, d[1]))
    Dp = U[mm][a] @ P[b] @ np.swapaxes(U[mm][a], -1, -2) - P[a]; Mst[a] += Dp @ Dp
    Dm = np.swapaxes(U[mm][a], -1, -2) @ P[a] @ U[mm][a] - P[b]; Mst[b] += Dm @ Dm
MU = float(os.environ.get('MU', 0))            # μ·nᵀΦ²n: вибирає вісь Φ з найменшим |власним значенням| (у трійці — e2), без домішки дублета
Peff = KAP*P - ETA*Mst + MU*(P @ P)
LQ = float(os.environ.get('QUAD', 0))           # λ(nᵀΦn)²: Φ безслідове, тож мінімум |nᵀΦn| — трійка, а в ній вісь із найменшим |власн. знач.|
def energy_grad(n):
    E = np.einsum('xyi,xyij,xyj', n, Peff, n); g = 2*np.einsum('xyij,xyj->xyi', Peff, n)
    if LQ:
        x = np.einsum('xyi,xyij,xyj->xy', n, P, n); E += LQ*np.sum(x**2); g += 4*LQ*x[..., None]*np.einsum('xyij,xyj->xyi', P, n)
    for mm, d in enumerate(G.DIRS):
        a = slice(max(0, -d[0]), WI - max(0, d[0])), slice(max(0, -d[1]), WJ - max(0, d[1]))
        b = slice(max(0, d[0]), WI + min(0, d[0])), slice(max(0, d[1]), WJ + min(0, d[1]))
        X = np.einsum('xyij,xyj->xyi', U[mm][a], n[b]); f = np.sum(n[a]*X, -1)
        E -= K/2*np.sum(f**2)
        g[a] -= K*f[..., None]*X
        g[b] -= K*f[..., None]*np.einsum('xyji,xyj->xyi', U[mm][a], n[a])
    return E, g
NIT = int(os.environ.get("NIT", 40000))
step = 0.2/(KAP + 6*K + 2*ETA*0.08 + 2*LQ + MU*0.3)
RP = float(os.environ.get('PIN', 0)); free = ((m['rw'] < RP) | (m['rw'] >= 20.5))[..., None] if RP else np.ones((WI, WJ, 1), bool)
vel = np.zeros_like(n)
for it in range(NIT + 1):
    E, g = energy_grad(n); gt = (g - np.sum(g*n, -1, keepdims=True)*n)*free; gn = np.sqrt(np.max(np.sum(gt**2, -1)))
    if it % 5000 == 0: print(it, f'{E:.8f}', f'{gn:.2e}', flush=True)
    if gn < 1e-7: break
    vel = 0.9*vel - step*gt; vel -= np.sum(vel*n, -1, keepdims=True)*n
    if np.sum(vel*gt) > 0: vel[:] = 0
    n = n + vel; n /= np.linalg.norm(n, axis=-1, keepdims=True)
print('E', E, 'it', it)
# орієнтація: |n·eo|² у рамці (далеко)
ov = []; rw = m['rw']
for (i, j), Fm in F.items():
    if rw[i, j] > 6: ov.append((Fm.T @ n[i, j])@eo)
ov = np.array(ov)**2; print('|n·e_odd|² далеко: mean', ov.mean().round(4), 'min', ov.min().round(4), 'max', ov.max().round(4))
np.savez(f"{D_}/dir_{OR}_k{K}" + (f"_eta{ETA}" if ETA else "") + (f"_q{LQ:g}" if LQ else "") + (f"_mu{MU:g}" if MU else "") + (f"_pin{RP:g}" if RP else "") + ("_fix" if NIT == 0 else "") + ".npz", n=n, E=E, K=K, KAP=KAP)
