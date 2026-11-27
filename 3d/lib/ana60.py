"""ana60 (№ 60): найбільші пласкі ділянки (|гол.| трикутників на вузлі < 0,35) і ψ у них — з кінцевого стану прогону rungen/rungen4."""
import numpy as np, sys, json
import fabgen as G, rnd51 as R
def flat_clusters(name, kind):
    d = np.load(name + ".npz"); f = G.FabricG(16, kind); R.flip(f, d["bad"]); z, psi = d["z"], d["psi"]
    hol = np.abs(f.holonomy(z)); hn = np.bincount(f.TV.ravel(), np.repeat(hol, 3), f.n)/f.tnode; M = hn < 0.35
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
    cs.sort(key=len, reverse=True)
    return dict(name=name, flat=float(M.mean()), sizes=[len(c) for c in cs[:5]], psi_big=[round(float(psi[c].mean()), 3) for c in cs[:3]], psi_all=float(psi.mean()))
if __name__ == "__main__":
    out = [flat_clusters(a.split(":")[0], a.split(":")[1]) for a in sys.argv[1:]]
    for o in out: print(o)
    json.dump(out, open("ana60.json", "w"))
