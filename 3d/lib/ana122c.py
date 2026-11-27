"""ana122c: чи сидять вихори на стінці між вакуумами Φ★/−Φ★. (а) кінцеві стани прогонів (120)/(122): вакуум-свідомий детектор
без «змішаних» трикутників (вершини в різних вакуумах) проти з ними; (б) знімки (122): частка вихорів ближче 1 до стінки."""
import json, glob, re, numpy as np, gf104 as GFm, matter98 as M, phi90 as PH, gauge91 as G
L = 96; X, Y = PH.coords(L); sh = G.sh
def key(f):
    m = re.search(r"_s(\d)_.*?(?:_b([\d.]+))?_pf", f); return f"{m.group(2) or '1.5'}_{m.group(1)}"
print("(а) кінцеві стани: вихорів усього / поза змішаними трикутниками; частка світу на стінці")
out = {}
for f in sorted(glob.glob("*_gk1_gd*_g104.npz")):
    if "gv100" in f: continue
    d = np.load(f); psi = d["psi"]; GF = GFm.GF(L, 1); GF.P = d["P"]; GF.U = d["U"]
    det, wm = GF.defects(psi, aware=True); npos = (np.linalg.eigvalsh(d["P"]) > 0).sum(-1) >= 3
    mixed = np.array([(npos != sh(npos, (1, 0))) | (npos != sh(npos, (0, 1))), (sh(npos, (1, 0)) != sh(npos, (1, 1))) | (sh(npos, (1, 1)) != sh(npos, (0, 1)))])
    n_all = len(M.clusters(np.where((det < 0) & wm, -1.0, 1.0), X, Y, L)); n_pure = len(M.clusters(np.where((det < 0) & wm & ~mixed, -1.0, 1.0), X, Y, L))
    out[key(f) + ("_sn" if "_sn_" in f else "")] = (n_all, n_pure, float((mixed & wm).sum()/max(wm.sum(), 1)))
for k in sorted(out): print(k, out[k][0], out[k][1], round(out[k][2], 3))
json.dump(out, open("ana122c.json", "w"))
