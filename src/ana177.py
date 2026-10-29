"""ana177: світи на дірці полотна. Для кожного прогону (TAG/s*_seed*_R*_q*): час першого світу, світи на дірці (вузол у колі R+2 від центру) і поза нею."""
import numpy as np, json, glob, os, sys, re
NB = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, 1), (1, -1)]
def dist(L):
    v = np.arange(L*L); x, y = v//L, v % L; X = x + 0.5*y; Y = np.sqrt(3)/2*y; c = (L//2 + 0.5*(L//2), np.sqrt(3)/2*(L//2)); d = None
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            e = np.hypot(X - c[0] + a*L + b*L/2, Y - c[1] + b*np.sqrt(3)/2*L); d = e if d is None else np.minimum(d, e)
    return d
def comps(B, L):
    lab = -np.ones(L*L, int); out = []
    for v in np.where(B)[0]:
        if lab[v] >= 0: continue
        st = [v]; lab[v] = len(out); mem = [v]
        while st:
            u = st.pop(); x, y = divmod(u, L)
            for dx, dy in NB:
                w = ((x + dx) % L)*L + (y + dy) % L
                if B[w] and lab[w] < 0: lab[w] = len(out); st.append(w); mem.append(w)
        out.append(np.array(mem))
    return [m for m in out if len(m) >= 5]
def one(d):
    R, q = map(float, re.search(r"_R([\d.]+)_q([\d.]+)", d).groups()); js = glob.glob(d + "/*_eg0.01.json"); sp = glob.glob(d + "/*_sp.npz")
    if not js or not sp: return None
    J = json.load(open(js[0])); L = int(re.search(r"_L(\d+)", js[0]).group(1)); D = dist(L); Rz = (R if R else 8) + 2
    t1 = next((w["t"] for w in J["w2"] if w["nuc"] > 0), None); S = np.load(sp[0]); rows = []
    for t, p in zip(S["t"], S["psi"]):
        cs = comps(p > 0, L); on = [c for c in cs if D[c].min() <= Rz]
        rows.append(dict(t=float(t), n=len(cs), on=len(on), on_size=int(sum(len(c) for c in on)), off_size=int(sum(len(c) for c in cs) - sum(len(c) for c in on))))
    first_on = next((r["t"] for r in rows if r["on"] > 0), None); first_any = next((r["t"] for r in rows if r["n"] > 0), None)
    return dict(run=d, R=R, q=q, L=L, t1=t1, first_on=first_on, first_any=first_any, final=rows[-1], rows=rows)
if __name__ == "__main__":
    out = [r for r in (one(d) for d in sorted(glob.glob(sys.argv[1] + "/s*"))) if r]
    json.dump(out, open(sys.argv[1] + "/ana177.json", "w"))
    for r in out: print(os.path.basename(r["run"]), "t1", r["t1"], "first_on", r["first_on"], "final n", r["final"]["n"], "on", r["final"]["on"], r["final"]["on_size"], "off", r["final"]["off_size"])
