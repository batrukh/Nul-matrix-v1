"""mass180: (180b) local "mass" of the gauge modes from the residual Φ on a real v1 snapshot (r173, stage B, T200).
The Higgs curvature of a link (2/3)|[G,Φ']|² is divided by the kinetic weight ka·w/2 → m² = (2/3)|[G,Φ']|²/(ka·w); the small weight of the Matrix ENHANCES the contribution of Φ.
We take the largest eigenvalue of the 10×10 matrix over the SO(5) generators (heaviest mode) and the smallest (lightest)."""
import numpy as np, sys, json
from scipy.ndimage import distance_transform_edt
from gauge91 import gen, DIRS, sh
f = sys.argv[1]; d = np.load(f); psi, P = d["psi"], d["P"]; L = psi.shape[0]
chi = 0.5*(1 + np.tanh(psi/0.2)); w = np.maximum(chi, 0.02)
Gs = [gen(i, j) for i in range(5) for j in range(i + 1, 5)]
out = {}
for m, dd in enumerate(DIRS):
    Pp = sh(P, dd); wl = 0.5*(w + sh(w, dd))
    C = np.array([G @ Pp - Pp @ G for G in Gs])                           # (10, L, L, 5, 5)
    M = np.einsum("a...ij,b...ij->...ab", C, C)                          # Gram
    ev = np.linalg.eigvalsh(M)                                           # (L, L, 10)
    out[m] = (2/3)*ev/(wl[..., None])
m2 = np.mean([out[m] for m in range(3)], axis=0)                         # mean over the three link directions
mat = psi < 0; dist = distance_transform_edt(mat); dw = distance_transform_edt(~mat)
res = dict(file=f.split("/")[-1], frac_matrix=float(mat.mean()))
rows = []
for lo, hi in [(0, 2), (2, 4), (4, 8), (8, 99)]:
    s = mat & (dist > lo) & (dist <= hi)
    if s.any():
        mx = m2[s][:, -1]; rows.append(dict(depth=f"{lo}-{hi}", n=int(s.sum()), m2_sorted_med=[float(x) for x in np.median(m2[s], axis=0)], m2_heavy_med=float(np.median(mx)), m2_heavy_p90=float(np.percentile(mx, 90)),
                                             m_heavy_med=float(np.sqrt(np.median(mx))), trP2_med=float(np.median(np.einsum("...ij,...ji", P, P)[s]))))
s = (~mat) & (dw > 3)
if s.any(): rows.append(dict(depth="world>3", n=int(s.sum()), m2_heavy_med=float(np.median(m2[s][:, -1])), m2_light_med=float(np.median(m2[s][:, 0])),
                             m2_sorted_med=[float(x) for x in np.median(m2[s], axis=0)]))
res["rows"] = rows; print(json.dumps(res, indent=1)); json.dump(res, open("mass180_" + res["file"].replace(".npz", ".json"), "w"))
# profile versus distance from the domain boundary (negative — inside the domain), medians of the 4 light and 6 heavy modes
sd = np.where(mat, dist, -dw); prof = []
for r in range(-6, 21):
    s = (np.round(sd) == r)
    if s.sum() >= 20: prof.append(dict(r=r, n=int(s.sum()), light=float(np.median(m2[s][:, :4].max(1))), heavy=float(np.median(m2[s][:, 4:].min(1))),
                                       heavy_p90=float(np.percentile(m2[s][:, -1], 90))))
res["prof"] = prof; json.dump(res, open("mass180_" + res["file"].replace(".npz", ".json"), "w"))
