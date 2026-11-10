"""(295) перевірка: проходження «фотона» крізь межу світу (181) при κ_b = 2 (база gf104) замість 1. python3 imp_kb.py FLOOR K DELTA KB"""
import sys, json; sys.path.insert(0, '/mnt/project-files/model/s52/r180'); sys.path.insert(0, '/mnt/project-files/model/s52/r268/code'); sys.path.insert(0, '/mnt/project-files/model/s52/r124')
import imp181
f, k, d, kb = map(float, sys.argv[1:5]); r = imp181.run(f, k, delta=d, kb=kb); fm = [1 - q['fw'] for q in r['rec']]
print(f, k, d, kb, 'imp', round(r['T_imp'], 4), 'fM', [round(fm[t-1], 4) for t in (40, 60, 80, 100)], flush=True)
json.dump(r, open(f'imp_f{f:g}_k{k:g}_d{d:g}_kb{kb:g}.json', 'w'))
