"""(343) крок 3: яка яма V (на центрі й першому кільці, rc=1) потрібна, щоб квант зв'язався: конус зі скрутом (непарний квант) проти пласкої без скруту (парний)."""
import json, numpy as np
from cone343 import spectrum
res = {}
for nsec, tw in ((5, 1), (5, 0), (6, 0), (7, 1)):
    for V in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
        e = [spectrum(nsec, R, tw, V=V)['E'][0] for R in (25, 35)]
        q = spectrum(nsec, 35, tw, V=V)
        res[f'n{nsec}_tw{tw}_V{V}'] = dict(E25=e[0], E35=e[1], bound=bool(e[1] < 0.1 - 1e-4 and abs(e[0] - e[1]) < 1e-3), core=q['core'])
        print(nsec, tw, V, [round(x, 5) for x in e], round(q['core'], 3), flush=True)
json.dump(res, open('well343.json', 'w'), indent=1)
