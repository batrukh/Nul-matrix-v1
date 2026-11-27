"""sum121: перерахунок (103) вакуум-свідомим детектором. Ті самі прогони local60 pf (без калібрувального поля), знімки раз на 5 од.
Старий детектор — matter98.holonomy (площина = 2 найбільші власні вектори); новий — у домені «три додатні» площина = пара від'ємних.
Також f3 — частка світу у вакуумі −Φ★."""
import json, numpy as np, kib103 as K, matter98 as M, phi90 as PH

def holo_aware(P):
    w, v = np.linalg.eigh(P); B = np.where(((w > 0).sum(-1) >= 3)[..., None, None], v[..., :, :2], v[..., :, 3:])
    tri = [((0, 0), (1, 0), (0, 1)), ((1, 0), (1, 1), (0, 1))]; sh = lambda A, d: np.roll(A, (-d[0], -d[1]), axis=(0, 1)); out = []
    for t in tri:
        b = [sh(B, d) for d in t]; T = np.eye(2)
        for k in range(3): T = M.polar2(np.swapaxes(b[(k + 1) % 3], -1, -2) @ b[k]) @ T
        out.append(np.linalg.det(T))
    return np.array(out)

B = "lc_s03.5_n100_L96_s"; E = "_tau8_D1_g0.02_iso_pf1"; L = 96; X, Y = PH.coords(L); res = {}
for s in (1, 2):
    for suf in ("", "p1", "e0.01", "e0.1"):
        nm = f"{B}{s}{E}{suf}"; t, psi, P, z = K.load(nm); r = dict(t=t.tolist(), old=[], new=[], f3=[], wtri=[])
        for k in range(len(t)):
            w = K.tri_world(psi[k], L); Pk = P[k].astype(float)
            d0, _ = M.holonomy(Pk); d1 = holo_aware(Pk)
            r["old"].append(len(M.clusters(np.where((d0 < 0) & w, -1.0, 1.0), X, Y, L)))
            r["new"].append(len(M.clusters(np.where((d1 < 0) & w, -1.0, 1.0), X, Y, L)))
            W = psi[k].reshape(L, L) > 0.3; r["f3"].append(float(((np.linalg.eigvalsh(Pk) > 0).sum(-1) >= 3)[W].mean()) if W.any() else None); r["wtri"].append(int(w.sum()))
        tt = np.array(t); o = np.array(r["old"]); n = np.array(r["new"])
        r.update(old_max=int(o.max()), new_max=int(n.max()), old_late=float(o[tt >= 100].mean()), new_late=float(n[tt >= 100].mean()),
                 f3_late=float(np.mean([f for f, q in zip(r["f3"], tt) if q >= 100 and f is not None])))
        res[nm] = r
        print(s, suf or "base", "max old/new", r["old_max"], r["new_max"], "| t>=100 old", round(r["old_late"], 1), "new", round(r["new_late"], 1), "| f3", round(r["f3_late"], 2), "| t5-20 old", r["old"][:4], "new", r["new"][:4], flush=True)
json.dump(res, open("sum121.json", "w"))
