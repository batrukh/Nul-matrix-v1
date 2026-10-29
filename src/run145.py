"""run145: пружне полотно (ela139) + довільні параметри local60.run, зокрема τ (ell у local60). Пружність: eg=γ, el=ℓ (базово 0,01 і 8)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ela139 as E
E.GAM, E.ELL = 0.01, 8.0
kw = {}
for a in sys.argv[1:]:
    k, v = a.split("=")
    if k == "eg": E.GAM = float(v)
    elif k == "el": E.ELL = float(v)
    else: kw[k] = int(v) if k in ("L", "n", "seed", "sk", "ln", "snp", "iso", "pseed", "gz", "gk", "gd", "sn") else float(v)
d = E.M.run(**kw); d["eg"], d["el"] = E.GAM, E.ELL
json.dump(d, open(d["name"] + f"_eg{E.GAM:g}.json", "w"))
print(d["name"], "final", d["final"][:12])
for q in d["w2"][::20]: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in q.items()})
