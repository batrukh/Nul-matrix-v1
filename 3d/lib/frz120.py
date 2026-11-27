"""frz120: (120) світ заморожено (ψ сталий, кінець живого прогону gf104), калібрувальне поле Φ★+U живе далі.
Режим plant=0: чи народжуються вихори без руху ψ. plant=1: посадка вихору в найглибшу точку світу, час життя.
Виклик: python3 frz120.py <stem>_g104.npz plant=0|1 T=60"""
import sys, json, numpy as np, scipy.ndimage as ndi
import gf104 as GFm, matter98 as M, phi90 as PH

def run(path, plant=0, T=60.0, dt=0.005, KG=10, seed=1, src=None, tag="", aware=False, b=0.0):
    """src: None — поле Φ★, U з того самого прогону; "fresh" — новий шум (як на старті); шлях — поле з іншого прогону (обмін)"""
    d = np.load(path); psi = d["psi"].astype(float); L = int(round(np.sqrt(psi.size)))
    G = GFm.GF(L, seed, b=b)                              # (123): b — кубічний член потенціалу Φ★
    if src not in ("fresh", "keepP", "keepU"):
        e = np.load(src) if src else d; G.P = e["P"].copy(); G.U = e["U"].copy()
    elif src == "keepP": G.P = d["P"].copy()              # власне Φ★, свіже U = 1
    elif src == "keepU": G.U = d["U"].copy()              # власне U, свіже Φ★
    G.Pi[:] = 0; G.E[:] = 0
    X, Y = PH.coords(L); dist = ndi.distance_transform_edt(np.tile(psi.reshape(L, L) > 0, (3, 3)))[L:2*L, L:2*L]
    pl = None
    if plant:
        i0, j0 = np.unravel_index(np.argmax(dist), dist.shape); pl = dict(ij=(int(i0), int(j0)), depth=float(dist[i0, j0]), xy=G.plant(i0, j0))
    tr = []; t = 0.0; n = int(round(1.0/(KG*dt)))
    for k in range(int(round(T))+1):
        det, wm = G.defects(psi, aware=aware); cl = M.clusters(np.where((det < 0) & wm, -1.0, 1.0), X, Y, L)
        if aware:                                         # (122): без «змішаних» трикутників (вершини в різних вакуумах Φ★/−Φ★) — стінка
            v3 = (np.linalg.eigvalsh(G.P) > 0).sum(-1) >= 3; sh_ = lambda A, d_: np.roll(A, (-d_[0], -d_[1]), (0, 1))
            mx = np.array([(v3 != sh_(v3, (1, 0))) | (v3 != sh_(v3, (0, 1))), (sh_(v3, (1, 0)) != sh_(v3, (1, 1))) | (sh_(v3, (1, 1)) != sh_(v3, (0, 1)))])
            npure = len(M.clusters(np.where((det < 0) & wm & ~mx, -1.0, 1.0), X, Y, L)); nwall = int((mx & wm).sum())
        near = None
        if pl and cl: near = round(min(float(M.wrapd(np.array(c[0]), np.array(c[1]), *pl["xy"], L)[0]) for c in cl), 1)
        w_ = psi.reshape(L, L) > 0.3
        tr.append(dict(t=float(k), ncl=len(cl), cl=cl, near=near, wtri=int(wm.sum()), kin=float((G.Pi**2).sum(axis=(-1, -2))[w_].mean()), el=float((G.E**2).sum()), t2=float(np.trace(G.P @ G.P, axis1=-2, axis2=-1)[w_].mean())))
        if aware: tr[-1].update(npure=npure, nwall=nwall)
        for _ in range(n): G.step(psi, KG*dt, 1.0)          # повернене ψ відкидаємо: світ заморожено
    out = dict(src=path, plant=pl, dmax=float(dist.max()), track=tr)
    json.dump(out, open(path.replace("_g104.npz", f"_frz120_p{plant}{tag}.json"), "w"))
    print(path, "plant", pl, "ncl", [r["ncl"] for r in tr][::5], "npure", [r.get("npure") for r in tr][::5], "nwall", [r.get("nwall") for r in tr][::10], "kin", [round(r["kin"], 5) for r in tr][::10], "t2", [round(r["t2"], 3) for r in tr][::10], "near", [r["near"] for r in tr][::5], flush=True)

if __name__ == "__main__":
    src = next((a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("src=")), None)
    kw = {k: float(v) for k, v in (a.split("=") for a in sys.argv[2:] if not a.startswith(("src=", "tag=")))}
    tag = next((a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("tag=")), "")
    run(sys.argv[1], plant=int(kw.get("plant", 0)), T=kw.get("T", 60.0), src=src, tag=tag, aware=bool(kw.get("aware", 0)), b=kw.get("b", 0.0))
