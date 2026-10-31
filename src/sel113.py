"""(113): what selects protodomains to become domains. Frames (111): 4 realisations of 2D L96, t = 1..60. A protodomain is a connected flat patch of the Matrix
(|hol.| < 0.35, ψ < 0, farther than 3 from domains), ≥ 3 nodes. Age — how many consecutive frames the patch existed (overlap ≥ 30 %).
"Became a domain" — if in the next 1..5 frames ψ > 0 on ≥ 3 of its nodes and this domain is new (did not touch old domains).
We compare the features in the last frame before birth with the same features of long-lived patches of the same age that did not become domains."""
import os; os.environ["NB"] = "0"
import numpy as np, json, local60 as LC, contextlib, io
from scipy.ndimage import distance_transform_edt as edt, label
with contextlib.redirect_stdout(io.StringIO()): from diag111 import node_feats
L = 96; STR = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]])
s = LC.Loc(L=L, h=0.3, lc=1.3); f = s.fab; t0, h0 = f.tail.copy(), f.head.copy()
def lab(M):
    lb, _ = label(np.tile(M, (3, 3)), structure=STR); return lb[L:2*L, L:2*L]
def auc(x, y):
    r = np.argsort(np.argsort(np.r_[x, y])) + 1; return float((r[:len(x)].sum() - len(x)*(len(x) + 1)/2)/(len(x)*len(y)))
rows = []
for sd in (1, 2, 3, 4):
    d = np.load(os.environ.get("SNPAT", "lc_s03.5_n100_L96_s{sd}_tau8_D1_g0.02_iso_sn_sn.npz").format(sd=sd)); PS = d["psi"]; T = len(d["t"])
    HH, AA, EX = [], [], []
    for k in range(T):
        tl = d["tail"][k]; fl = tl != t0; f.tail = tl; f.head = np.where(fl, t0, h0)
        A, H = node_feats(f, d["z"][k].astype(complex)); HH.append(H); AA.append(A)
        od = np.bincount(tl, None, f.n); EX.append((od == 0) | (od == 6))             # extremum of τ: all arrows of a node point outwards or inwards
    W = [(PS[k] > 0).reshape(L, L) for k in range(T)]
    far = [edt(~np.tile(W[k], (3, 3)))[L:2*L, L:2*L] > 3 for k in range(T)]
    FL = [(HH[k].reshape(L, L) < 0.35) & (PS[k].reshape(L, L) < 0) & far[k] for k in range(T)]
    LB = [lab(F_) for F_ in FL]
    age = [np.zeros((L, L), int) for _ in range(T)]
    for k in range(1, T):                                                               # age: inherited from the previous frame through overlap
        for c in np.unique(LB[k])[1:]:
            m = LB[k] == c; prev = age[k - 1][m & FL[k - 1]]
            age[k][m] = (prev.max() + 1) if (m & FL[k - 1]).sum() >= 0.3*m.sum() else 1
    for k in range(10, T - 5):
        for c in np.unique(LB[k])[1:]:
            m = LB[k] == c
            if m.sum() < 3: continue
            born = False
            for j in range(k + 1, k + 6):
                newW = W[j] & ~W[j - 1]
                if (newW & m).sum() >= 3:
                    lbw = lab(W[j]); ids = np.unique(lbw[newW & m]); ids = ids[ids > 0]
                    if any(not (W[j - 1] & (lbw == i)).any() for i in ids): born = True; break
            mm = m.reshape(-1)
            rows.append(dict(seed=sd, t=k, born=born, size=int(m.sum()), age=int(age[k][m].max()), ext=int(EX[k][mm].sum()),
                             extfrac=float(EX[k][mm].mean()), psi=float(PS[k][mm].mean()), A2=float(AA[k][mm].mean()), hol=float(HH[k][mm].mean())))
json.dump(rows, open("sel113.json", "w"))
B = [r for r in rows if r["born"]]; N_ = [r for r in rows if not r["born"]]
print("patch-instances", len(rows), "born", len(B), "unique births", len({(r["seed"], r["t"]) for r in B}))
for q in ("size", "age", "ext", "extfrac", "psi", "A2", "hol"):
    x = np.array([r[q] for r in B]); y = np.array([r[q] for r in N_]); yl = np.array([r[q] for r in N_ if r["age"] >= 3])
    print(f"{q:8s} born med {np.median(x):7.3f}  not-born med {np.median(y):7.3f}  long-lived not-born med {np.median(yl):7.3f}  AUC {auc(x, y):.2f}  AUC vs long {auc(x, yl):.2f}")
