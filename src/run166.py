"""run166: matter with elasticity (28.10). Full live model (Φ★ + SO(5) gauge field, vacuum-aware detector, as rungd (120)) + elastic canvas ela139.
Elasticity keys: ge= (γ), le= (ℓ); the rest are parameters of local60.run (ell=, g= here — pre-time τ and tilt, as in (120)). ck= — checkpoint (time units)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ela139 as E, local60 as M
kw = {}; E.GAM = 0.01; E.ELL = 8.0
for a in sys.argv[1:]:
    k, v = a.split("=")
    if k == "ge": E.GAM = float(v)
    elif k == "le": E.ELL = float(v)
    elif k == "ckdir": kw[k] = v
    else: kw[k] = int(v) if k in ("L", "n", "seed", "sk", "ln", "snp", "iso", "gz", "gk", "gd", "sn", "ck") else float(v)
r = M.run(**kw)
tag = f"ge{E.GAM:g}_le{E.ELL:g}"
json.dump(r, open(os.path.basename(r["name"]) + f"_{tag}.json", "w"))
print("done", r["name"], tag, flush=True)
