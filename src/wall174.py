"""wall166: де сидять вихори кінця прогону (166) — відстань до стінки Φ★|−Φ★ у світі (як (122)); контроль — випадкові вузли світу."""
import sys, os, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/.."); import phi90 as PH, matter98 as M98
L = 96; X, Y = PH.coords(L); rng = np.random.default_rng(0); out = {}
for d in sorted(glob.glob(os.path.dirname(os.path.abspath(__file__)) + "/b*_s*")):
    z = np.load(glob.glob(d + "/*_g104.npz")[0]); tr = json.load(open(glob.glob(d + "/*_g104.json")[0]))["track"]
    v3 = (np.linalg.eigvalsh(z["P"]) > 0).sum(-1) >= 3; W = z["psi"].reshape(L, L) > 0
    wall = np.zeros((L, L), bool)
    for sh in ((1, 0), (0, 1), (-1, 1)):
        dif = v3 != np.roll(v3, sh, (0, 1)); wall |= dif & (W | np.roll(W, sh, (0, 1))); wall |= np.roll(dif & (W | np.roll(W, sh, (0, 1))), (-sh[0], -sh[1]), (0, 1))
    wx, wy = X[wall], Y[wall]
    def dw(x, y): return float(M98.wrapd(wx, wy, x, y, L)[0].min()) if wall.any() else np.inf
    cl = [c for r in tr if r["t"] >= 200 for c in r["cl"]]
    dv = [dw(c[0], c[1]) for c in cl]
    ii = rng.choice(np.where(W.ravel())[0], 300); dr = [dw(X.ravel()[k], Y.ravel()[k]) for k in ii]
    out[os.path.basename(d)] = dict(nvort=len(dv), near3=round(float(np.mean(np.array(dv) < 3)), 2) if dv else None, rand3=round(float(np.mean(np.array(dr) < 3)), 2), wallnodes=int(wall.sum()))
    print(os.path.basename(d), out[os.path.basename(d)])
json.dump(out, open("wall174.json", "w"))
