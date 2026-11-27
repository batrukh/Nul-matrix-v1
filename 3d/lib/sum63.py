import numpy as np, json, chir63 as C
for tag in ["", "_ac0", "_ac-1"]:
    for sd in (1, 2):
        fn = f"lc_s03.5_n100_L96_s{sd}_tau8{tag}"
        f = C.fab_tau(96, 8, sd); d = np.load(fn + ".npz"); z, psi = d["z"], d["psi"]
        l1, l2, l3 = f.tri_links(z); s = np.sin(np.angle(l1*l2*l3)); sig = np.sign(f.S1 + f.S2 + f.S3)
        B = psi > 0; inw = B[f.TV].all(1); out = (~B[f.TV]).all(1)
        w = json.load(open(fn + ".json"))["w2"]; q = [x for x in w if x["t"] > 120]
        print(tag or "_ac1", sd, "worlds", round(np.mean([x["nuc"] for x in q]), 1), "nB", round(np.mean([x["nB"] for x in q])),
              "boil", round(np.mean([x["boil"] for x in q if "boil" in x]), 2),
              "| <s*sig> in", round((s*sig)[inw].mean(), 4) if inw.any() else None, "out", round((s*sig)[out].mean(), 4), "| <|s|> out", round(np.abs(s)[out].mean(), 3))
