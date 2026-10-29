"""run166: речовина з пружністю (28.10). Повна жива модель (Φ★ + калібр. SO(5), вакуум-свідомий детектор, як rungd (120)) + пружне полотно ela139.
Ключі пружності: ge= (γ), le= (ℓ); решта — параметри local60.run (ell=, g= тут — до-час τ і нахил, як у (120)). ck= — контрольна точка (од. часу)."""
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
