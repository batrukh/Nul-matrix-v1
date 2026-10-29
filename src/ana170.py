"""(170) протосвіти 3D: розміри зв'язних пласких інтенсивних ділянок (|гол.| < 0,35, |z|² > 5) і ψ у них; L16 (117) проти L24 (170)."""
import os; os.environ["NB"] = "0"
import numpy as np, json, sys, fabgen as G
def ana(path, L):
    f = G.FabricG(L, "tet"); d = np.load(path); z, psi = d["z"], d["psi"]
    hol = np.abs(f.holonomy(z)); hn = np.bincount(f.TV.ravel(), np.repeat(hol, 3), f.n)/f.tnode
    A2n = (np.bincount(f.tail, np.abs(z)**2, f.n) + np.bincount(f.head, np.abs(z)**2, f.n))/f.deg; M = (hn < 0.35) & (A2n > 5)
    nb = [[] for _ in range(f.n)]
    for p, q in zip(f.tail, f.head): nb[p].append(q); nb[q].append(p)
    seen = np.zeros(f.n, bool); cs = []
    for v in np.flatnonzero(M):
        if seen[v]: continue
        comp = [v]; seen[v] = True; st = [v]
        while st:
            u = st.pop()
            for w in nb[u]:
                if M[w] and not seen[w]: seen[w] = True; comp.append(w); st.append(w)
        cs.append(comp)
    cs.sort(key=len, reverse=True); big = cs[0] if cs else []
    return dict(path=os.path.basename(path), L=L, proto=float(M.mean()), ncl=len(cs), top=[len(c) for c in cs[:5]],
                big_frac=len(big)/f.n, psi_big=float(psi[big].mean()) if big else None, psi_max=float(psi.max()))
if __name__ == "__main__":
    out = [ana(p, int(p.split("_L")[1].split("_")[0])) for p in sys.argv[1:]]
    for o in out: print(o)
    json.dump(out, open("ana170.json", "w"))
