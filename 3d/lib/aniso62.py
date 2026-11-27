"""aniso62: № 62 крок 1 — чи світи орієнтовані за напрямком нахилу τ. Декартове вкладення трикутної ґратки: e1=(1,0), e2=(1/2,√3/2).
Для кожного світу (≥ 30 вузлів): тензор інерції розгорнутих координат → кут головної осі (mod 180°) і видовження λ1/λ2.
Нахил τ = g·(x + 0,31·y) в індексах → градієнт w = (1, −0,219), кут −12,4°."""
import numpy as np, os, glob
os.environ.setdefault("DELTA", "1e-6")
import simNlib as N
L = 96; f = N.Fabric(L); E1 = np.array([1.0, 0.0]); E2 = np.array([0.5, np.sqrt(3)/2])
nb = [[] for _ in range(f.n)]
for p, q in zip(f.tail, f.head): nb[p].append(q); nb[q].append(p)
def dvec(u, w):
    d = f.xy[w] - f.xy[u]; d = (d + L//2) % L - L//2; return d[0]*E1 + d[1]*E2
def worlds(psi, mn=30):
    B = psi > 0; seen = np.zeros(f.n, bool); out = []
    for v in np.where(B)[0]:
        if seen[v]: continue
        pos = {v: np.zeros(2)}; st = [v]; seen[v] = True; wrap = False
        while st:
            u = st.pop()
            for w in nb[u]:
                if not B[w]: continue
                pw = pos[u] + dvec(u, w)
                if w in pos:
                    if np.abs(pos[w] - pw).max() > 1e-6: wrap = True
                else: pos[w] = pw; seen[w] = True; st.append(w)
        if len(pos) >= mn:
            P = np.array(list(pos.values())); P -= P.mean(0); C = P.T @ P/len(P); ev, vec = np.linalg.eigh(C)
            ang = np.degrees(np.arctan2(vec[1, 1], vec[0, 1])) % 180
            out.append((len(pos), round(float(ang), 1), round(float(np.sqrt(ev[1]/max(ev[0], 1e-9))), 2), wrap, C/np.trace(C)))
    return out
TILT = np.degrees(np.arctan2(-0.2194, 1.0)) % 180
groups = {"нахил (загоєний)": ["lc_s03.5_n100_L96_s1_tau8_D1_g0.01.npz", "bak68/lc_s03.5_n100_L96_s1_tau8_D1_g0.02.npz", "bak68/lc_s03.5_n100_L96_s2_tau8_D1_g0.02.npz",
                                "bak59/lc_s03.5_n100_L96_s2_tau8_D1_g0.02.npz", "bak71/lc_s03.5_n100_L96_s1_tau8_D1_g0.04.npz", "lc_s03.5_n100_L96_s1_tau8_D1_g0.04_T1200_cut433.npz",
                                "lc_s03.5_n100_L96_s2_tau8_D1_g0.02_T2100_cut313.npz"],
          "без нахилу, нерухоме τ (66)": [f"lc_s03.5_n100_L96_s{s}_tau8.npz" for s in (1, 2, 3, 4)],
          "вихори + нахил (77)": sorted(glob.glob("lc_s03.5_n100_L96_s*_g0.02*_v*g0.02.npz"))}
print("кут нахилу (mod 180):", round(TILT, 1))
for gname, fs in groups.items():
    allw = []
    for fn in fs:
        w = worlds(np.load(fn)["psi"]); allw += w
        print("  ", fn.split("/")[-1], [(a, b, c, "WRAP" if d else "") for a, b, c, d, _ in w])
    if not allw: continue
    wts = np.array([x[0] for x in allw], float); Cm = sum(x[0]*x[4] for x in allw)/wts.sum(); ev, vec = np.linalg.eigh(Cm)
    angs = np.radians(2*np.array([x[1] for x in allw])); R_ = np.hypot((wts*np.cos(angs)).sum(), (wts*np.sin(angs)).sum())/wts.sum()
    mang = np.degrees(np.arctan2((wts*np.sin(angs)).sum(), (wts*np.cos(angs)).sum()))/2 % 180
    print(gname, "світів", len(allw), "середня вісь", round(float(mang), 1), "узгодженість осей R =", round(float(R_), 2), "(0 — хаос, 1 — усі паралельні)",
          "сер. видовження", round(float(np.median([x[2] for x in allw])), 2))
