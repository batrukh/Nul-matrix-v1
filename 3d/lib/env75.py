"""env75: довкілля обрізаного світу в момент обрізання (№ 61 / (75)): чому в реалізації 1 світ 50–70 гине, а в 2 росте."""
import numpy as np, os, json
os.environ.setdefault("DELTA", "1e-6")
import simNlib as N, rnd51 as R, local60 as A
L = 96
def replay(seed, g, Tt, D=1, ell=8, dt=0.005):
    f = N.Fabric(L); tau = A.tau_field(L, ell, seed)
    x_, y_ = f.xy[f.tail].T; x2, y2 = f.xy[f.head].T
    Gg = g*(((x2-x_+L//2) % L-L//2)+0.31*((y2-y_+L//2) % L-L//2))
    bad = tau[f.head]-tau[f.tail]+Gg < 0; R.flip(f, bad); Gg = np.where(bad, -Gg, Gg)
    for _ in range(int(round(Tt/(200*dt)))):
        ns = int(np.ceil(12*D))
        for _ in range(ns):
            g_ = tau[f.head]-tau[f.tail]; tau = tau+(D*200*dt/ns)*(np.bincount(f.tail, g_, f.n)-np.bincount(f.head, g_, f.n))
        fl = tau[f.head]-tau[f.tail]+Gg < 0; R.flip(f, fl); Gg = np.where(fl, -Gg, Gg)
    return f, tau, Gg
def comps(f, nb, B):
    lab = -np.ones(f.n, int); k = 0
    for v in np.where(B)[0]:
        if lab[v] >= 0: continue
        st = [v]; lab[v] = k
        while st:
            u = st.pop()
            for w in nb[u]:
                if B[w] and lab[w] < 0: lab[w] = k; st.append(w)
        k += 1
    return lab
def dist(nb, src, n):
    d = -np.ones(n, int); d[src] = 0; q = list(src)
    while q:
        q2 = []
        for u in q:
            for w in nb[u]:
                if d[w] < 0: d[w] = d[u]+1; q2.append(w)
        q = q2
    return d
def env(f, tau, Gg, z, psi, pick):
    nb = [[] for _ in range(f.n)]
    for p, q in zip(f.tail, f.head): nb[p].append(q); nb[q].append(p)
    B = psi > 0; lab = comps(f, nb, B); sizes = np.bincount(lab[lab >= 0])
    c = pick(sizes); W = lab == c; d = dist(nb, list(np.where(W)[0]), f.n)
    Aa = np.abs(z)**2; de = np.maximum(d[f.tail], d[f.head]); grad = np.abs(tau[f.head]-tau[f.tail]+Gg)
    others = [s for i, s in enumerate(sizes) if i != c and s >= 5]
    dmin = min((d[lab == i].min() for i, s in enumerate(sizes) if i != c and s >= 5), default=None)
    l1, l2, l3 = f.tri_links(z); sh = np.sin(np.angle(l1*l2*l3)); tv = d[f.TV].max(1)
    out = dict(size=int(W.sum()), Ain=round(float(Aa[de == 0].mean()), 2), psi_in=round(float(psi[W].mean()), 3))
    for r in ((1, 3), (4, 8), (9, 15)):
        m = (de >= r[0]) & (de <= r[1]); mn = (d >= r[0]) & (d <= r[1])
        out[f"A{r[0]}-{r[1]}"] = round(float(Aa[m].mean()), 2); out[f"psi{r[0]}-{r[1]}"] = round(float(psi[mn].mean()), 3)
    m = de <= 8; out["grad"] = round(float(grad[m].mean()), 4); out["grad_all"] = round(float(grad.mean()), 4)
    out["tau_pct"] = round(float((tau < tau[W].mean()).mean()), 2); out["hol"] = round(float(np.abs(sh[tv <= 8]).mean()), 3)
    out["others"] = others; out["dmin"] = int(dmin) if dmin is not None else None; out["Aout_all"] = round(float(Aa[(psi[f.tail] < 0) & (psi[f.head] < 0)].mean()), 3)
    return out
big = lambda s: int(np.argmax(s))
for seed, g, T, tag in ((1, 0.04, 600, "R1"), (2, 0.02, 1500, "R2")):
    f, tau, Gg = replay(seed, g, T)
    for c in (100, 70, 50):
        d = np.load(f"lc_s03.5_n100_L96_s{seed}_tau8_D1_g{g:g}_T{T}_cut{c}.npz")
        # обрізаний світ — той, що перетинає колишній великий світ (у R1 при cut 50 він ~ розміром з природний)
        d0 = np.load(f"bak71/lc_s03.5_n100_L96_s1_tau8_D1_g0.04.npz" if seed == 1 else "bak59/lc_s03.5_n100_L96_s2_tau8_D1_g0.02.npz")
        nb = [[] for _ in range(f.n)]
        for p, q in zip(f.tail, f.head): nb[p].append(q); nb[q].append(p)
        lab0 = comps(f, nb, d0["psi"] > 0); bigW = lab0 == np.argmax(np.bincount(lab0[lab0 >= 0]))
        lab = comps(f, nb, d["psi"] > 0); cid = np.bincount(lab[bigW & (lab >= 0)]).argmax()
        print(tag, "cut", c, env(f, tau, Gg, d["z"], d["psi"], lambda s: cid), flush=True)
    d0 = np.load("bak71/lc_s03.5_n100_L96_s1_tau8_D1_g0.04.npz" if seed == 1 else "bak59/lc_s03.5_n100_L96_s2_tau8_D1_g0.02.npz")
    print(tag, "uncut big", env(f, tau, Gg, d0["z"], d0["psi"], big))
    if seed == 1:
        print(tag, "natural small", env(f, tau, Gg, d0["z"], d0["psi"], lambda s: int(np.argsort(s)[-2])))
