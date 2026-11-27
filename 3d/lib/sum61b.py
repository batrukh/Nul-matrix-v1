import json, os
fs = [f"lc_s03.5_n100_L96_s1_tau8_D1_g0.04_T1200_cut{c}.json" for c in (70, 60, 40, 30)] + [f"lc_s03.5_n100_L96_s2_tau8_D1_g0.02_T2100_cut{c}.json" for c in (313, 200, 100, 70, 50)]
for fn in fs:
    if not os.path.exists(fn): continue
    d = json.load(open(fn)); w = d["w2"]
    print(fn, "final", d["final"][:6])
    print("   ", [(x["t"], x["big"], x["nuc"]) for x in w if x["t"] % 25 == 0])
