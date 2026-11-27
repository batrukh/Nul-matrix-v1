import json
for c in (433, 200, 100, 50):
    d = json.load(open(f"lc_s03.5_n100_L96_s1_tau8_D1_g0.04_T1200_cut{c}.json")); w = d["w2"]
    print(c, "final", d["final"][:6])
    print("   ", [(x["t"], x["big"], x["nuc"], round(x["boil"], 2) if "boil" in x else None) for x in w if x["t"] % 25 == 0 or x["t"] < 611])
