# (315) аналіз: світи, кипіння, ψ на аморфній Матриці
import json, glob, os, sys, numpy as np
sys.path.insert(0, "/mnt/project-files/model/s52/r268/code"); import rnd51 as R
out = {}
for d in sorted(glob.glob("/mnt/project-files/model/s52/r315/j*")):
    js = glob.glob(d + "/*_r315.json")
    if not js: continue
    D = json.load(open(js[0])); w = D["w2"]; nm = os.path.basename(d)
    jit = float(nm.split("_")[0][1:]); g = int(nm.split("_g")[1]) if "_g" in nm else 0
    irr = 0.0 if jit == 0 else float((R.RandFabric(96, g, jit).deg != 6).mean())
    psi = np.load(glob.glob(d + "/lc_*.npz")[0])["psi"]
    t = [q["t"] for q in w]; nB = [q["nB"] for q in w]
    late = [q for q in w if q["t"] >= 200]
    out[nm] = dict(jit=jit, irr=round(100*irr, 1), nB_end=nB[-1], frac_end=round(nB[-1]/9216, 3), worlds_end=sum(1 for c in D["final"] if c >= 5),
                   big_end=D["final"][0] if D["final"] else 0, first_t=next((q["t"] for q in w if q["nuc"] > 0), None),
                   boil=round(float(np.mean([q["boil"] for q in late])), 3), sd=round(float(np.mean([q["sd"] for q in late])), 2),
                   psimax=round(float(psi.max()), 2), psimean=round(float(psi.mean()), 3), t=t, nB=nB)
    print(nm, {k: v for k, v in out[nm].items() if k not in ("t", "nB")})
json.dump(out, open("/mnt/project-files/model/s52/r315/ana315.json", "w"))
