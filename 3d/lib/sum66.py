"""sum66: № 66 — порівняння (66)–(70) з ізотропним τ проти старого (анізотропного)."""
import json, os, glob
pairs = [("(66) τ s%d" % s, f"lc_s03.5_n100_L96_s{s}_tau8") for s in (1, 2, 3, 4)] + \
        [("(67) D1 s%d" % s, f"lc_s03.5_n100_L96_s{s}_tau8_D1") for s in (1, 2)] + \
        [("(68) g0.02 s%d" % s, f"lc_s03.5_n100_L96_s{s}_tau8_D1_g0.02") for s in (1, 2)] + \
        [("(70) rnd%s" % r, f"lc_s03.5_n100_L96_s1_tau8_rnd{r}g1") for r in ("0.3", "0.45", "0.5", "1")]
old = {"(67) D1 s1": "bak67/lc_s03.5_n100_L96_s1_tau8_D1", "(67) D1 s2": "bak67/lc_s03.5_n100_L96_s2_tau8_D1",
       "(68) g0.02 s1": "bak68/lc_s03.5_n100_L96_s1_tau8_D1_g0.02", "(68) g0.02 s2": "bak68/lc_s03.5_n100_L96_s2_tau8_D1_g0.02"}
def summ(fn):
    if not os.path.exists(fn + ".json"): return None
    d = json.load(open(fn + ".json")); w = d["w2"]; last = w[-1]; tail = [x for x in w if x["t"] >= last["t"] - 50]
    return dict(t=last["t"], worlds=last["nuc"], sizes=d["final"][:6], share=round(sum(x["nB"] for x in tail)/len(tail)/9216, 3),
                boil=round(sum(x.get("boil", 0) for x in tail)/len(tail), 2))
for lab, base in pairs:
    print(lab, "\n   старе:", summ(old.get(lab, base)), "\n   iso:  ", summ(base + "_iso"))
