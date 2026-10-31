"""prof188: (188) profile of the domain boundary in model v1 with the field Φ★ (r173 stage B, L96, 4 runs pf1/pf0.3 × s1/s2, frames t = 100–200).
For each node — signed distance to the boundary (> 0 in a domain, < 0 in the Matrix; distance to the nearest node of the opposite phase, on the torus).
Averages in distance bands: ψ, |z|² of the node's links, |hol.| of triangles, tr Φ★². Writes prof188.json."""
import os, sys; os.environ["NB"] = "0"; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, json, contextlib, io, local60 as LC
from scipy.spatial import cKDTree
L = 96; B = os.environ.get("R188_SNAP", "light_pf{pf}_s{sd}/lc_s03.5_n100_L96_s{sd}_sk4l0_pf{pf}_snap.npz")
with contextlib.redirect_stdout(io.StringIO()): f = LC.Loc(L=L, h=0.3, lc=1.3).fab
i_, j_ = np.divmod(np.arange(L*L), L); XY = np.c_[i_ + j_/2, j_*np.sqrt(3)/2]
SH = [(p*L + q*L/2, q*np.sqrt(3)/2*L) for p in (-1, 0, 1) for q in (-1, 0, 1)]
def dist_to(mask):                                                          # distance from each node to the nearest node in mask (torus)
    pts = np.vstack([XY[mask] + s for s in SH]); return cKDTree(pts).query(XY)[0]
def feats(z, psi, P):
    a_, b_, n = f.tail, f.head, f.n; A2 = np.abs(z)**2; cnt = np.bincount(a_, None, n) + np.bincount(b_, None, n)
    A = (np.bincount(a_, A2, n) + np.bincount(b_, A2, n))/cnt; hol = np.abs(f.holonomy(z)); tn = np.zeros(n); tc = np.zeros(n)
    for j in range(3): np.add.at(tn, f.TV[:, j], hol); np.add.at(tc, f.TV[:, j], 1)
    t2 = np.einsum("xyij,xyji->xy", P, P).reshape(-1); return dict(psi=psi, A2=A, hol=tn/tc, trP2=t2)
bins = np.arange(-12.0, 7.1, 1.0); cen = 0.5*(bins[1:] + bins[:-1]); out = {}
for pf in ("1", "0.3"):
    for sd in (1, 2):
        d = np.load(B.format(pf=pf, sd=sd)); acc = {k: [[] for _ in cen] for k in ("psi", "A2", "hol", "trP2")}
        for k in np.flatnonzero(d["t"] >= 100):
            psi = d["psi"][k].astype(float); W = psi > 0
            if W.sum() < 20: continue
            sdist = np.where(W, dist_to(~W) - 0.5, -(dist_to(W) - 0.5)); sdist = np.where(W, np.maximum(sdist, 0.01), np.minimum(sdist, -0.01)); q = feats(d["z"][k].astype(complex), psi, d["P"][k])
            ib = np.digitize(sdist, bins) - 1
            for key in acc:
                for b in range(len(cen)): acc[key][b].extend(q[key][ib == b].tolist())
        out[f"pf{pf}_s{sd}"] = {key: [float(np.mean(v)) if len(v) else None for v in acc[key]] for key in acc}
        out[f"pf{pf}_s{sd}"]["n"] = [len(v) for v in acc["psi"]]; print(pf, sd, flush=True)
json.dump(dict(cen=cen.tolist(), runs=out), open("prof188.json", "w"))
