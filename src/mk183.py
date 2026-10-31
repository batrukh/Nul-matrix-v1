"""mk183: (183) states for the intervention in the island. For each event (seed, frame k = j − 3) — a disc of radius R around the centre of the new domain (frame j).
Variants: ctl — no change (reproducibility check); scr — link phases in the disc randomised (|z| the same, budget unchanged);
pn — phase noise σ = 0.5 rad; sham — the same scr in a disc of the same radius elsewhere in the Matrix (farther than 3 from domains, no birth in frames k..k+15).
Writes rs/<seed>_k<k>_<variant>[_<event>].npz (z, ψ, t) and rs/events.json."""
import numpy as np, json, os, sys
from scipy.ndimage import distance_transform_edt as edt
L = 96; R = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
SN = os.environ.get("R183_SN", "birth_s{sd}/lc_s03.5_n100_L96_s{sd}_sk4l0_sn_sn.npz")  # frames from birth.sh
ev = [e for e in json.load(open("ev183_dk3.json")) if e["t_j"] <= 22.5]
i_, j_ = np.divmod(np.arange(L*L), L); X = i_ + j_/2; Y = j_*np.sqrt(3)/2            # node coordinates (i + j/2, j·√3/2)
def tdist(x0, y0):
    best = None
    for p in (-1, 0, 1):
        for q in (-1, 0, 1):
            d = np.hypot(X - x0 - p*L - q*L/2, Y - y0 - q*np.sqrt(3)/2*L); best = d if best is None else np.minimum(best, d)
    return best
def centre(nodes):                                                                  # centre on the torus via circular means
    ang = lambda v, P: np.angle(np.mean(np.exp(2j*np.pi*v/P)))*P/(2*np.pi) % P
    jj = ang(j_[nodes], L); ii = ang(i_[nodes], L); return ii + jj/2, jj*np.sqrt(3)/2
from local60 import Loc
f = Loc(L=L, h=0.3, lc=1.3).fab                                                     # sk=4: lattice arrows, tails are not flipped
os.makedirs("rs", exist_ok=True); rng = np.random.default_rng(183); out = []; cache = {}
for n_, e in enumerate(ev):
    sd, k = e["seed"], e["k"]
    if sd not in cache: cache[sd] = np.load(SN.format(sd=sd))
    d = cache[sd]; z = d["z"][k].astype(complex); psi = d["psi"][k].astype(float); t = float(np.round(d["t"][k]))
    x0, y0 = centre(np.array(e["new"])); disk = tdist(x0, y0) <= R
    lk = disk[f.tail] & disk[f.head]
    base = f"{sd}_k{k}"
    if not os.path.exists(f"rs/{base}_ctl.npz"): np.savez(f"rs/{base}_ctl.npz", z=z, psi=psi, t=t)
    zs = np.where(lk, np.abs(z)*np.exp(2j*np.pi*rng.random(f.m)), z); np.savez(f"rs/{base}_scr_e{n_}.npz", z=zs, psi=psi, t=t)
    zp = np.where(lk, z*np.exp(1j*0.5*rng.standard_normal(f.m)), z); np.savez(f"rs/{base}_pn_e{n_}.npz", z=zp, psi=psi, t=t)
    W = [(d["psi"][q] > 0) for q in range(k, min(k + 16, len(d["t"])))]; ever = np.any(W, 0)      # domain at any time in k..k+15
    farw = edt(~np.tile(ever.reshape(L, L), (3, 3)))[L:2*L, L:2*L].reshape(-1) > R + 3
    cand = np.flatnonzero(farw & (psi < 0))
    c = cand[rng.integers(len(cand))]; sdisk = tdist(X[c], Y[c]) <= R; slk = sdisk[f.tail] & sdisk[f.head]
    zh = np.where(slk, np.abs(z)*np.exp(2j*np.pi*rng.random(f.m)), z); np.savez(f"rs/{base}_sham_e{n_}.npz", z=zh, psi=psi, t=t)
    out.append(dict(ev=n_, seed=sd, k=k, t_k=t, t_j=e["t_j"], nnew=e["nnew"], cx=float(x0), cy=float(y0), R=R, nlinks=int(lk.sum()),
                    sham=[float(X[c]), float(Y[c])], disk=np.flatnonzero(disk).tolist(), sdisk=np.flatnonzero(sdisk).tolist()))
    print(n_, base, "t", t, "→", e["t_j"], "links", int(lk.sum()), flush=True)
json.dump(out, open("rs/events.json", "w"))
