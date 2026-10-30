"""ph145: «фотон» на кінцевому стані живого прогону (як (92)), старт у найглибшій точці найбільшого світу.
Пишемо частку енергії хвилі в усіх світах (fin) і у власному світі (fown) — з пружністю світи дрібні й сусідні."""
import numpy as np, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gauge91 as G, scipy.ndimage as nd
from phi90 import coords
src = sys.argv[1]; floor = float(sys.argv[2]); kind = sys.argv[3] if len(sys.argv) > 3 else "photon"
psi = np.load(src)["psi"]; L = psi.shape[0]; B = psi > 0
st = np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]])                       # сусіди трикутної ґратки (±(1,0), ±(0,1), ±(−1,1))
lab, n = nd.label(np.tile(B, (3, 3)), structure=st); lab = lab[L:2*L, L:2*L]
sz = np.bincount(lab.ravel()); sz[0] = 0; big = int(np.argmax(sz))
dist = nd.distance_transform_edt(np.tile(B, (3, 3)))[L:2*L, L:2*L]
i0, j0 = np.unravel_index(np.argmax(np.where(lab == big, dist, -1)), dist.shape)
X, Y = coords(L); G.OWN = lab == big
if hasattr(G, "PARTIAL"): G.PARTIAL = "part_" + os.path.basename(src).replace("_g91.npz", "") + f"_f{floor:g}.json"
name, rec, dE = G.run(src=src, var="world", kind=kind, floor=floor, ka=1.0, kb=2.0, T=float(os.environ.get("PH_T", "40")), cen=f"{X[i0, j0]},{Y[i0, j0]}")
q = rec[-1]
out = dict(src=src, floor=floor, kind=kind, size=int((lab == big).sum()), depth=float(dist[i0, j0]), nworld=int(B.sum()), fin=q["fin"], fown=q["fown"], drift=dE,
           fin_t=[(r["t"], round(r["fin"], 4), round(r["fown"], 4)) for r in rec])
tag = os.path.basename(src).replace("_g91.npz", "") + f"_{kind}_f{floor:g}"
json.dump(out, open(f"ph182_{tag}_T{os.environ.get("PH_T", "40")}.json", "w")); print(tag, "size", out["size"], "depth", round(out["depth"], 1), "fin %.3f fown %.3f drift %.1e" % (q["fin"], q["fown"], dE))
