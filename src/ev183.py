"""ev183: (183) domain births in the frames of (173) stage A (v1, L96, s0 3.5, 4 seeds) and the preceding islands.
An island is a connected flat patch of the Matrix (|hol| < 0.35, ψ < 0, farther than 3 from domains), as in sel113. For each new domain (frame j)
we take frame k = j − DK and the island that overlaps the nodes of the new domain. Writes ev183.json: seed, k, j, island nodes, nodes of the new domain."""
import os; os.environ["NB"] = "0"
import numpy as np, json, sys, local60 as LC, contextlib, io
from scipy.ndimage import distance_transform_edt as edt, label
def node_feats(f, z):                                                        # as diag111.node_feats
    n = f.n; a_, b_ = f.tail, f.head; A2 = np.abs(z)**2; cnt = np.bincount(a_, None, n) + np.bincount(b_, None, n)
    A = (np.bincount(a_, A2, n) + np.bincount(b_, A2, n))/cnt; hol = np.abs(f.holonomy(z)); tn = np.zeros(n); tc = np.zeros(n)
    for j in range(3): np.add.at(tn, f.TV[:, j], hol); np.add.at(tc, f.TV[:, j], 1)
    return A, tn/tc
L = 96; STR = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]]); DK = int(sys.argv[1]) if len(sys.argv) > 1 else 3
s = LC.Loc(L=L, h=0.3, lc=1.3); f = s.fab; t0, h0 = f.tail.copy(), f.head.copy()
def lab(M):
    lb, _ = label(np.tile(M, (3, 3)), structure=STR); return lb[L:2*L, L:2*L]
R = os.environ.get("R183_SN", "birth_s{sd}/lc_s03.5_n100_L96_s{sd}_sk4l0_sn_sn.npz")   # frames of birth.sh
ev = []
for sd in (1, 2, 3, 4):
    d = np.load(R.format(sd=sd)); PS = d["psi"]; T = len(d["t"]); W = [(PS[k] > 0).reshape(L, L) for k in range(T)]
    for j in range(DK, min(T, 40)):
        lbw = lab(W[j]); newW = W[j] & ~W[j - 1]
        for c in np.unique(lbw[newW]):
            if c == 0 or (W[j - 1] & (lbw == c)).any(): continue          # only a truly new domain (not growth of an old one)
            nw = lbw == c
            if nw.sum() < 3: continue
            k = j - DK; tl = d["tail"][k]; fl = tl != t0; f.tail = tl; f.head = np.where(fl, t0, h0)
            A, H = node_feats(f, d["z"][k].astype(complex))
            far = edt(~np.tile(W[k], (3, 3)))[L:2*L, L:2*L] > 3
            FL = (H.reshape(L, L) < 0.35) & (PS[k].reshape(L, L) < 0) & far; LB = lab(FL)
            ids, cnt = np.unique(LB[nw], return_counts=True); cnt = cnt[ids > 0]; ids = ids[ids > 0]
            isl = (LB == ids[np.argmax(cnt)]) if len(ids) else np.zeros((L, L), bool)
            ev.append(dict(seed=sd, k=k, j=j, t_k=float(d["t"][k]), t_j=float(d["t"][j]), nnew=int(nw.sum()), nisl=int(isl.sum()),
                           isl=np.flatnonzero(isl).tolist(), new=np.flatnonzero(nw).tolist()))
            print(sd, "k", k, "t", d["t"][k], "→", d["t"][j], "new", int(nw.sum()), "island", int(isl.sum()), flush=True)
json.dump(ev, open(f"ev183_dk{DK}.json", "w"))
