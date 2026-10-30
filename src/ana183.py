"""ana183: (183) чи народився світ у місці події після втручання. Зона — вузли на відстані ≤ R + 2 від центру нового світу.
Світ «народився», якщо в якомусь кадрі t_k+1..t_k+15 у зоні ≥ 3 вузлів з ψ > 0. Для sham — те саме і в зоні підміненого диска."""
import numpy as np, json, os, sys
P = os.environ.get("R183_DIR", "."); L = 96
i_, j_ = np.divmod(np.arange(L*L), L); X = i_ + j_/2; Y = j_*np.sqrt(3)/2
def tdist(x0, y0):
    best = None
    for p in (-1, 0, 1):
        for q in (-1, 0, 1):
            d = np.hypot(X - x0 - p*L - q*L/2, Y - y0 - q*np.sqrt(3)/2*L); best = d if best is None else np.minimum(best, d)
    return best
def born(fn, zone):
    if not os.path.exists(fn): return None
    a = np.load(fn); W = a["psi"].astype(float) > 0; c = (W[:, zone]).sum(1)
    k = np.flatnonzero(c >= 3); return (float(a["t"][k[0]]) if len(k) else None), int(c.max())
ev = json.load(open(f"{P}/rs/events.json")); rows = []
for e in ev:
    z = tdist(e["cx"], e["cy"]) <= e["R"] + 2; zs = tdist(*e["sham"]) <= e["R"] + 2; b = f"{P}/psi/{e['seed']}_k{e['k']}"
    r = dict(ev=e["ev"], seed=e["seed"], t_k=e["t_k"], t_j=e["t_j"])
    for v, fn in (("ctl", f"{b}_ctl.npz"), ("scr", f"{b}_scr_e{e['ev']}.npz"), ("pn", f"{b}_pn_e{e['ev']}.npz"), ("sham", f"{b}_sham_e{e['ev']}.npz")):
        q = born(fn, z); r[v] = None if q is None else q[0]; r[v + "_ok"] = q is not None
        if v == "sham" and q is not None: r["sham_site"] = born(fn, zs)[0]
    rows.append(r)
json.dump(rows, open(f"{P}/ana183.json", "w"))
done = [r for r in rows if r["ctl_ok"] and r["scr_ok"] and r["pn_ok"] and r["sham_ok"]]
print("events complete", len(done), "of", len(rows))
for v in ("ctl", "scr", "pn", "sham"):
    nb = sum(r[v] is not None for r in done); dl = [r[v] - r["t_j"] for r in done if r[v] is not None]
    print(f"{v:5s} born at site {nb:2d}/{len(done)}  delay vs original med {np.median(dl) if dl else float('nan'):+.1f}")
print("sham site born", sum(r.get("sham_site") is not None for r in done), "/", len(done))
for r in done: print(r["ev"], r["seed"], r["t_k"], r["t_j"], "ctl", r["ctl"], "scr", r["scr"], "pn", r["pn"], "sham", r["sham"], "shamsite", r.get("sham_site"))
