"""sym62: чи рівняння руху симетричні щодо повороту ґратки на 120° і 60° (перевірка апарату, № 62)."""
import numpy as np, os
os.environ.setdefault("DELTA", "1e-6")
import local60 as A, rnd51 as R
L = 12
def rot_map(f, k):
    # поворот на 120° (k=1) або 60° (k=2): індекси (i,j) → нові індекси; e1=(1,0), e2=(1/2,√3/2)
    i, j = f.xy.T
    if k == 1: ni, nj = (-i - j) % L, i % L              # R120: e1 → e2 − e1, e2 → −e1
    else: ni, nj = (i - j) % L, (i) % L                   # R60: e1 → e2, e2 → e2 − e1
    if k == 2: ni, nj = (-j) % L, (i + j) % L
    nid = {tuple(p): n for n, p in enumerate(f.xy.tolist())}
    return np.array([nid[(a, b)] for a, b in zip(ni, nj)])
def edge_index(f):
    return {(a, b): e for e, (a, b) in enumerate(zip(f.tail.tolist(), f.head.tolist()))}
rng = np.random.default_rng(3)
for k, nm in ((1, "120°"), (2, "60°")):
    s1 = A.Loc(L=L, h=0.3, lc=1.3, ac=1.0); f1 = s1.fab
    R.flip(f1, rng.random(f1.m) < 0.3)                    # довільні стріли
    z = 1.8*(1 + 0.3*(rng.standard_normal(f1.m) + 1j*rng.standard_normal(f1.m))); psi = 0.3*rng.standard_normal(f1.n)
    dz1, dp1 = super(A.Loc, s1).F(z, psi)
    nm_ = rot_map(f1, k)
    s2 = A.Loc(L=L, h=0.3, lc=1.3, ac=1.0); f2 = s2.fab
    E2 = {}
    for e, (a, b) in enumerate(zip(f2.tail.tolist(), f2.head.tolist())): E2[(a, b)] = (e, 1); E2[(b, a)] = (e, -1)
    emap = np.zeros(f1.m, int); flip = np.zeros(f2.m, bool)
    for e, (a, b) in enumerate(zip(f1.tail.tolist(), f1.head.tolist())):
        e2, sg = E2[(nm_[a], nm_[b])]; emap[e] = e2; flip[e2] = sg < 0
    R.flip(f2, flip)                                      # стріли повернуто разом із ґраткою
    z2 = np.zeros(f2.m, complex); z2[emap] = z; p2 = np.zeros(f2.n); p2[nm_] = psi
    dz2, dp2 = super(A.Loc, s2).F(z2, p2)
    print(nm, "max |Δ ż| =", float(np.abs(dz2[emap] - dz1).max()), " max |Δ ψ̇| =", float(np.abs(dp2[nm_] - dp1).max()), " |ż| ~", float(np.abs(dz1).mean()))
