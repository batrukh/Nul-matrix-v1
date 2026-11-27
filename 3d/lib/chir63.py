"""chir63: «закрученість» (хіральність) світів — черга № 63. Голономія трикутника h = arg(l1·l2·l3) (усі трикутники обходяться
проти год. стрілки); фізична величина, не залежить від того, як записано стріли. Для кожного світу (зв'язна група ψ>0 ≥ 5 вузлів):
середнє sin h по трикутниках, усі вершини яких у світі; так само поза світами (ψ<0)."""
import numpy as np, sys, os
os.environ.setdefault("DELTA", "1e-6")
import simNlib as N, rnd51 as R, local60 as A, birth48 as BR

def fab_tau(L, ell, seed):
    f = N.Fabric(L); tau = A.tau_field(L, ell, seed); R.flip(f, tau[f.head] < tau[f.tail]); return f

def chir(f, z, psi):
    l1, l2, l3 = f.tri_links(z); h = np.angle(l1*l2*l3); s = np.sin(h)
    nb = [[] for _ in range(f.n)]
    for p, q in zip(f.tail, f.head): nb[p].append(q); nb[q].append(p)
    B = psi > 0; lab = -np.ones(f.n, int); k = 0
    for v in np.where(B)[0]:
        if lab[v] >= 0: continue
        st = [v]; lab[v] = k; comp = [v]
        while st:
            u = st.pop()
            for w in nb[u]:
                if B[w] and lab[w] < 0: lab[w] = k; st.append(w); comp.append(w)
        k += 1
    TL = lab[f.TV]; out = (~B[f.TV]).all(1)
    res = []
    for c in range(k):
        m = (TL == c).all(1); size = int((lab == c).sum())
        if size >= 5 and m.sum() > 0: res.append((size, round(float(s[m].mean()), 4), int(m.sum())))
    return sorted(res, reverse=True), round(float(s[out].mean()), 4), round(float(s.std()), 3)

if __name__ == "__main__":
    for sd in (1, 2, 3, 4):
        f = fab_tau(96, 8, sd); d = np.load(f"lc_s03.5_n100_L96_s{sd}_tau8.npz")
        w, o, sdv = chir(f, d["z"], d["psi"]); print("tau8 s", sd, "out", o, "sd", sdv, "worlds", w)
