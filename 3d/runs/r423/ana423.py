import json, glob, numpy as np
P = {(3.0,0):(0,8),(3.0,0.05):(8,8),(3.0,0.1):(16,16),(3.0,0.15):(8,8),(3.0,0.2):(13,16),(3.0,0.25):(1,8),(3.0,0.3):(0,1),(2.9,0.2):(7,8),(3.45,0.2):(0,8)}
out = []
for (s0, a), (k, n) in P.items():
    R = [json.load(open(f)) for f in sorted(glob.glob(f"runs/fdt423_s{s0:g}_ac{a:g}_seed*.json"))]
    if not R: continue
    T3 = np.array([r["Delta"][-1]/r["chi1"][-1] for r in R]); T1 = np.array([r["Delta"][3]/r["chi1"][3] for r in R])
    TA = np.array([r["DeltaA"][-1]/r["chiA1"][-1] if abs(r["chiA1"][-1]) > 1e-12 else np.nan for r in R])
    lin = np.array([r["chi2"][-1]/r["chi1"][-1] for r in R])
    d = dict(s0=s0, a=a, P=f"{k}/{n}", p=k/n, n=len(R), T3=T3.mean(), T3e=T3.std(ddof=1)/np.sqrt(len(R)) if len(R) > 1 else 0, T1=T1.mean(), T1e=T1.std(ddof=1)/np.sqrt(len(R)) if len(R) > 1 else 0,
             var=np.mean([r["var"] for r in R]), chi=np.mean([r["chi1"][-1] for r in R]), TA=float(np.nanmean(TA)), lin=[lin.min(), lin.max()], psim=np.mean([r["psimean"] for r in R]))
    out.append(d); print(f"s0 {s0} α {a}: P {k}/{n}  T3 {d['T3']:.2e}±{d['T3e']:.1e}  T1 {d['T1']:.2e}±{d['T1e']:.1e}  var {d['var']:.2e} χ {d['chi']:.3f} ψ̄ {d['psim']:.3f} lin {lin.min():.4f}-{lin.max():.4f} n{len(R)}")
hi = [d for d in out if d["p"] >= 0.75]; lo = [d for d in out if d["p"] <= 1/8 + 1e-9 and d["a"] != 0.3]
T1_, T2_ = min(d["T3"] for d in hi), max(d["T3"] for d in hi)
print("window", T1_, T2_)
for d in lo:
    gap = max(T1_ - d["T3"], d["T3"] - T2_); err = d["T3e"] + max(x["T3e"] for x in hi)
    print(d["s0"], d["a"], "gap", gap, "err", err, "gap/err", gap/err if err else None)
json.dump(out, open("ana423.json", "w"), default=float, indent=0)
