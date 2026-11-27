import json, numpy as np, kib103 as K
B="lc_s03.5_n100_L96_s"; E="_tau8_D1_g0.02_iso_pf1"; res={}
for s in (1,2):
    for suf in ("", "p1", "e0.01", "e0.1"):
        nm=f"{B}{s}{E}{suf}"; out, rows = K.analyse(nm)
        n=np.array([o["ncl"] for o in out]); w=np.array([o["wtri"] for o in out]); t=np.array([o["t"] for o in out])
        res[nm]=dict(t=t.tolist(), ncl=n.tolist(), wtri=w.tolist(), births=len(rows), cl=[o["cl"] for o in out])
        print(s, suf or "base", "max", n.max(), "at t", t[n.argmax()], "| t10", n[1], "| mean t>=100", round(n[t>=100].mean(),1), "| births", len(rows), "| world tri end", w[-1])
json.dump(res, open("sum103.json","w"))
