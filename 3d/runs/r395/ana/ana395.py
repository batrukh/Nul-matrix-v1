"""(395) аналіз: тор без меж, α 0,2, s0 3,45, 8 зерен, T300 (rq383g). Критерій — r395/criterion395.txt."""
import ast, json, glob, os
R = "/mnt/project-files/model/s52/r395"; out = []
for s in range(1, 9):
    rows = [ast.literal_eval(l) for l in open(f"{R}/runs/s345_seed{s}/log") if l.startswith("{")]
    t = [r["t"] for r in rows]; big = [r["big"] for r in rows]; A2 = [r["A2"] for r in rows]
    born = next((r["t"] for r in rows if r["big"] >= 50), None)
    out.append(dict(seed=s, tend=t[-1], big_end=big[-1], big_max=max(big), tborn=born, A2_end=A2[-1],
                    A2_min=min(A2), A2_max=max(A2), nB_max=max(r["nB"] for r in rows), nuc_max=max(r["nuc"] for r in rows),
                    boil_end=rows[-1]["boil"], flat_end=rows[-1]["flat"], t=t, big=big, A2=A2))
n = sum(o["big_end"] >= 50 for o in out)
v = "ПІДТВЕРДЖЕНО" if n <= 2 else ("СПРОСТОВАНО" if n >= 6 else "НЕЧІТКО")
for o in out: print(o["seed"], o["tend"], "big end/max", o["big_end"], o["big_max"], "born", o["tborn"], "A2", round(o["A2_min"],4), round(o["A2_end"],4), "nB", o["nB_max"], "nuc", o["nuc_max"], "flat", o["flat_end"])
print(f"світів {n}/8 → {v}")
json.dump(dict(n=n, verdict=v, runs=out), open(f"{R}/ana/ana395.json", "w"))
