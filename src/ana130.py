"""ana130: intensity budget → boiling and domains in a defect-free Matrix (sec. 4.4 (130)). local60 sk=4, L96, n=100, T=250, 2 realisations.
Averages over t ≥ 150: boiling, Matrix fraction in domains, number of domains (≥ 20 nodes), largest, intensity in and outside domains; time of the first nucleus and of boiling onset."""
import numpy as np, json, glob
import ana53 as A, birth48 as BR

out = []
for fn in sorted(glob.glob("lc_s0*_n100_L96_s*_sk4l0.json")):
    d = json.load(open(fn)); nm = d["name"]; s0 = float(nm.split("_s0")[1].split("_")[0]); seed = int(nm.split("_L96_s")[1].split("_")[0])
    f = A.fabric(np.load(nm + ".npz")["bad"]); nb = [[] for _ in range(f.n)]
    for p, q in zip(f.tail, f.head): nb[p].append(q); nb[q].append(p)
    sp = np.load(nm + "_sp.npz"); sel = np.where(sp["t"] >= 150 - 1e-6)[0]
    nw, big, msz = [], [], []
    for k in sel:
        cs = [c for c in BR.comps(f, sp["psi"][k] > 0, nb)] if (sp["psi"][k] > 0).any() else []
        w = [c for c in cs if c >= 20]; nw.append(len(w)); big.append(max(w) if w else 0); msz.append(np.mean(w) if w else 0)
    late = [r for r in d["w2"] if r["t"] >= 150]
    t_boil = next((r["t"] for r in d["w2"] if r.get("boil", 0) > 0.1), None); t_nuc = next((r["t"] for r in d["w2"] if r["nuc"] > 0), None)
    mean = lambda k: round(float(np.mean([r[k] for r in late if r.get(k) is not None])), 3) if any(r.get(k) is not None for r in late) else None
    out.append(dict(s0=s0, seed=seed, boil=mean("boil"), share=round(100*np.mean([r["nB"] for r in late])/96**2, 1), worlds=round(float(np.mean(nw)), 2),
                    big=round(float(np.mean(big))), msize=round(float(np.mean(msz))), Ain=mean("Ain"), Aout=mean("Aout"), t_boil=t_boil, t_nuc=t_nuc))
    print(out[-1])
json.dump(sorted(out, key=lambda r: (r["s0"], r["seed"])), open("ana130.json", "w"), indent=1)
