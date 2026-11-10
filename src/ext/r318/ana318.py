# (318) аналіз: найсильніші 3N ниток — чи це тріангуляція з 6 сусідами? Порівняння з Делоне тих самих точок.
import sys, json, numpy as np
from scipy.spatial import Delaunay
D = json.load(open(sys.argv[1])); I = np.array(D["I"]); E = np.array(D["E"]); P = np.array(D["P"]); n = len(P); L = D["args"]["L"]; H = L*np.sqrt(3)/2
top = np.argsort(-I)[:3*n]; k = np.bincount(E[top].ravel(), minlength=n)
img = np.concatenate([P + np.array([dx*L, dy*H]) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]); tri = Delaunay(img).simplices
c = img[tri].mean(1); keep = (c[:, 0] >= 0) & (c[:, 0] < L) & (c[:, 1] >= 0) & (c[:, 1] < H); t = tri[keep] % n
de = {tuple(sorted((int(a), int(b)))) for q in t for a, b in ((q[0], q[1]), (q[1], q[2]), (q[0], q[2]))}
sel = {tuple(sorted(map(int, E[e]))) for e in top}
r = dict(I_ratio=float(np.percentile(I, 90)/max(np.percentile(I, 10), 1e-12)), f6_top=float((k == 6).mean()), kdist_top=np.bincount(k).tolist(),
         overlap_delaunay=len(sel & de)/len(de), f6_delaunay=float((np.bincount(np.array(list(de)).ravel(), minlength=n) == 6).mean()),
         Ilen=[float(np.corrcoef(I, np.linalg.norm(((P[E[:, 0]] - P[E[:, 1]] + [L/2, H/2]) % [L, H]) - [L/2, H/2], axis=1))[0, 1])])
print(r)
