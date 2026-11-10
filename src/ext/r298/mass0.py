"""(298) етап 1: маси при k=0 у вакуумі комплексного Φ★ (μ=0,3, b=0). Гессіан V у 28-вимірному просторі (дійсні + уявні напрямки) і маси калібрувальних полів (2/3)|δΦ|²/k."""
import numpy as np, json
from vac import V, G, sym0
from itertools import combinations
mu = 0.3; lam = None
P = np.diag([-1, -1, -1, 1.5, 1.5]).astype(complex)
for _ in range(60000): P = P - 0.02*G(P, mu=mu)
lam = np.diag(P).real; print('vacuum', np.round(lam, 5), 'V', V(P, mu=mu))
# ортонормований базис безслідових симетричних (14) × {1, i}
B = []
for i in range(5):
    for j in range(i + 1, 5):
        X = np.zeros((5, 5)); X[i, j] = X[j, i] = 1/np.sqrt(2); B.append(X)
for k in range(4):
    v = np.zeros(5); v[:k + 1] = 1; v[k + 1] = -(k + 1); v /= np.linalg.norm(v); B.append(np.diag(v))
B = [b.astype(complex) for b in B] + [1j*b for b in B]
n = len(B); Hm = np.zeros((n, n)); h = 1e-4
for a in range(n):
    for b in range(n):
        f = lambda x, y: V(P + x*B[a] + y*B[b], mu=mu)
        Hm[a, b] = (f(h, h) - f(h, -h) - f(-h, h) + f(-h, -h))/(4*h*h)
w, v = np.linalg.eigh(Hm)
lab = []
for k in range(n):
    re = np.sum(v[:14, k]**2); lab.append('Re' if re > 0.5 else 'Im')
print('Φ modes ω² (Re/Im):'); [print(f'  {w[k]:.4f} {lab[k]}') for k in range(n)]
# калібрувальні маси
def Gen(i, j): X = np.zeros((5, 5)); X[i, j], X[j, i] = 1, -1; return X
ka, kc, q = 1., 1., 1.
gm = {f'G{i}{j}': (2/3)*np.sum(np.abs(Gen(i, j) @ P - P @ Gen(i, j))**2)/(ka*2) * 2 for i, j in combinations(range(5), 2)}
# |G|^2 = 2, кінетична ka/4|E|^2 = ka/2 ε̇²  →  ω² = (2/3)|[G,Φ]|²/ka
gm = {k: (2/3)*np.sum(np.abs(Gen(int(k[1]), int(k[2])) @ P - P @ Gen(int(k[1]), int(k[2])))**2)/ka for k in gm}
mB = (2/3)*q**2*np.sum(np.abs(P)**2)/kc
print('gauge ω²:', {k: round(v_, 4) for k, v_ in gm.items()}, ' U(1) B:', round(mB, 4))
json.dump({'vac': lam.tolist(), 'phi_w2': w.tolist(), 'phi_lab': lab, 'gauge_w2': gm, 'B_w2': mB}, open('mass0.json', 'w'), indent=1)
