"""ana120: народження і смерть Z₂-вихорів у живій моделі (детектор щоодиниці, gk=1) і рух ψ навколо місць народження.
Народження — вихор без попередника ближче r0 у попередньому записі; смерть — без наступника. Час життя — ланцюжки найближчих сусідів.
«Мерехтіння» ψ: вузли світу, де ψ за одиницю перетнуло 0 (межа світу) або впало нижче 0,3 (поріг детектора)."""
import sys, json, glob, numpy as np, scipy.ndimage as ndi
import phi90 as PH, matter98 as M
L = 96; X, Y = PH.coords(L); r0 = 2.5
def dd(p, q): return float(M.wrapd(np.array(p[0]), np.array(p[1]), q[0], q[1], L)[0])
def node(p): return int(np.argmin((X - p[0])**2 + (Y - p[1])**2))

def one(gj, t0=30.0):
    tr = json.load(open(gj))["track"]; sn = np.load(gj.replace("_g104.json", "_sn.npz"))
    ts = sn["t"]; PS = sn["psi"].reshape(len(ts), -1)
    births, deaths, life, bpsi, rpsi = [], 0, [], [], []
    alive = {}                                            # індекс вихору → час народження
    rng = np.random.default_rng(0)
    area = 0.0; nsum = 0.0; flick = 0.0; flick0 = 0.0
    for a, c in zip(tr, tr[1:]):
        if c["t"] < t0: continue
        dt = c["t"] - a["t"]; area += c["wtri"]*dt; nsum += len(a["cl"])*dt
        k = int(np.argmin(abs(ts - c["t"]))); kp = max(k - 1, 0)
        p1, p0 = PS[k], PS[kp]; W = p1 > 0.3
        flick += float(((p0 > 0.3) != (p1 > 0.3)).sum()); flick0 += float(((p0 > 0) != (p1 > 0)).sum())
        for p in c["cl"]:
            if all(dd(p, q) > r0 for q in a["cl"]):
                n_ = node(p); births.append(c["t"])
                bpsi.append([float(PS[max(k - m, 0)][n_]) for m in range(4)])     # ψ у місці народження: зараз, −1, −2, −3
        for p in a["cl"]:
            if all(dd(p, q) > r0 for q in c["cl"]): deaths += 1
        idx = np.flatnonzero(W)
        for n_ in rng.choice(idx, min(5, len(idx)), replace=False) if len(idx) else []:
            rpsi.append([float(PS[max(k - m, 0)][n_]) for m in range(4)])
    T = tr[-1]["t"] - t0; bp = np.array(bpsi); rp = np.array(rpsi)
    return dict(file=gj, births=len(births), deaths=deaths, birth_rate=1000*len(births)/area if area else None,
                meanN=nsum/T, life_little=nsum/max(deaths, 1), meanA=area/T,
                flick_rate=1000*flick/area, flick0_rate=1000*flick0/area,
                birth_psi=bp.mean(0).round(3).tolist() if len(bp) else None, rand_psi=rp.mean(0).round(3).tolist(),
                birth_psi_min3=float(np.mean(bp[:, 1:].min(1) < 0.3)) if len(bp) else None, rand_psi_min3=float(np.mean(rp[:, 1:].min(1) < 0.3)))

if __name__ == "__main__":
    out = {}
    for gj in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "*_gk1_g104.json")):
        r = one(gj); out[gj] = r; print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    json.dump(out, open("ana120.json", "w"))
