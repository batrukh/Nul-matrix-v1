"""(112): двостійкість ниток Матриці у 2D (гіпотеза після (111)). Знімки local60 sn=1, 4 реалізації, t = 10..60.
Вузли Матриці (ψ < −0,3), далі d від усіх світів. (а) розподіл у площині «закрученість |гол.| — інтенсивність |z|²»;
(б) дрейф: середня зміна закрученості за 1 од. часу залежно від поточної (нулі з від'ємним нахилом — стійкі стани, з додатним — поріг);
(в) пласкі ділянки (|гол.| < 0,35, ≥ 5 вузлів): скільки живуть і чим закінчуються (світ або розчинення)."""
import os; os.environ["NB"] = "0"
import numpy as np, json, local60 as LC
from scipy.ndimage import distance_transform_edt as edt, label
from diag111 import node_feats
L = 96; ST = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]])         # сусіди трикутної ґратки в індексах (i, j)
s = LC.Loc(L=L, h=0.3, lc=1.3); f = s.fab; t0, h0 = f.tail.copy(), f.head.copy()
def lab_periodic(M):
    lb, n = label(np.tile(M, (3, 3)), structure=ST); c = lb[L:2*L, L:2*L]; return c
out = dict(hist=None, drift={}, patches=[])
Hs, As = [], []; Hb = np.linspace(0, np.pi, 41); Ab = np.linspace(0, 12, 41); H2 = np.zeros((40, 40))
dh_sum = np.zeros(40); dh_n = np.zeros(40); dh5_sum = np.zeros(40); dh5_n = np.zeros(40)
for sd in (1, 2, 3, 4):
    d = np.load(os.environ.get("SNPAT", "lc_s03.5_n100_L96_s{sd}_tau8_D1_g0.02_iso_sn_sn.npz").format(sd=sd)); PS = d["psi"]; T = len(d["t"])
    HH = []; AA = []
    for k in range(T):
        tl = d["tail"][k]; fl = tl != t0; f.tail = tl; f.head = np.where(fl, t0, h0)
        A, H = node_feats(f, d["z"][k].astype(complex)); HH.append(H); AA.append(A)
    HH = np.array(HH); AA = np.array(AA)
    far = np.array([edt(~np.tile((PS[k] > 0).reshape(L, L), (3, 3)))[L:2*L, L:2*L].reshape(-1) > 4 for k in range(T)])
    M = (PS < -0.3) & far
    for k in range(10, T):
        H2 += np.histogram2d(HH[k][M[k]], AA[k][M[k]], [Hb, Ab])[0]
        if k + 1 < T:
            m = M[k] & M[k + 1]; b = np.clip(np.digitize(HH[k][m], Hb) - 1, 0, 39)
            dh_sum += np.bincount(b, HH[k + 1][m] - HH[k][m], 40); dh_n += np.bincount(b, None, 40)
        if k + 5 < T:
            m = M[k]; b = np.clip(np.digitize(HH[k][m], Hb) - 1, 0, 39)
            dh5_sum += np.bincount(b, HH[k + 5][m] - HH[k][m], 40); dh5_n += np.bincount(b, None, 40)
    # (в) пласкі ділянки Матриці: життя і доля
    flat = [((HH[k] < 0.35) & (PS[k] < 0) & far[k]).reshape(L, L) for k in range(T)]
    labs = [lab_periodic(F_) for F_ in flat]
    for k in range(10, T - 1):
        for c in np.unique(labs[k]):
            if c == 0: continue
            m = labs[k] == c
            if m.sum() < 5: continue
            if k > 10 and (flat[k - 1] & m).sum() >= 0.5*m.sum(): continue   # беремо лише нові ділянки
            life = 0; fate = "open"; mm = m
            for j in range(k + 1, T):
                if (PS[j].reshape(L, L)[mm] > 0).mean() > 0.5: fate = "world"; break
                ov = flat[j] & mm
                if ov.sum() < 0.3*mm.sum(): fate = "dissolve"; break
                life = j - k; mm = ov | (flat[j] & np.isin(labs[j], np.unique(labs[j][ov])[1:]))
            out["patches"].append(dict(seed=sd, t=k, size=int(m.sum()), life=life, fate=fate, A2=float(AA[k][m.reshape(-1)].mean())))
cen = (Hb[:-1] + Hb[1:])/2
out["hist"] = dict(H=H2.tolist(), Hb=Hb.tolist(), Ab=Ab.tolist())
out["drift"] = dict(h=cen.tolist(), d1=(dh_sum/np.maximum(dh_n, 1)).tolist(), n1=dh_n.tolist(), d5=(dh5_sum/np.maximum(dh5_n, 1)).tolist())
json.dump(out, open("bist112.json", "w"))
pH = H2.sum(1); print("marginal |hol| (bins of 0.0785):", np.round(pH/pH.sum()*100, 1).tolist())
print("drift 1:", [(round(h, 2), round(x, 3), int(n)) for h, x, n in zip(cen, out["drift"]["d1"], dh_n) if n > 50])
P = out["patches"]; from collections import Counter
print("patches", len(P), Counter(p["fate"] for p in P), "life median", np.median([p["life"] for p in P]) if P else None)
for fate in ("world", "dissolve", "open"):
    q = [p for p in P if p["fate"] == fate]
    if q: print(fate, "n", len(q), "size med", np.median([p["size"] for p in q]), "life med", np.median([p["life"] for p in q]), "A2 med", round(float(np.median([p["A2"] for p in q])), 2))
