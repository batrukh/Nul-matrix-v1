import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = json.load(open('drop345.json')); sP = 1.155*2*np.sqrt(np.pi*751)          # σ = (2√2/3)·√1,5 на трикутній ґратці, P = 2√(πA)
meas = {5: R['5']['centre']['E'] - R['5']['far']['E'], 7: R['7']['centre']['E'] - R['7']['far']['E'], 6: R['6']['centre']['E'] - R['6']['far']['E']}
pred = {n: sP*(np.sqrt(n/6) - 1) for n in (5, 6, 7)}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
a = ax[0]; x = np.arange(3); ns = [5, 6, 7]
a.bar(x - 0.18, [meas[n] for n in ns], 0.36, color='#d9480f', label='модель (крапля світу, 751 вузол)')
a.bar(x + 0.18, [pred[n] for n in ns], 0.36, color='#2a6fdb', label='геометрія: σ·P·(√(n/6) − 1)')
for i, n in enumerate(ns): a.text(i - 0.18, meas[n] + (0.6 if meas[n] >= 0 else -1.4), f'{meas[n]:+.1f}', ha='center', fontsize=9)
a.axhline(0, color='k', lw=0.6); a.set_xticks(x); a.set_xticklabels(['конус (5)', 'пласка (6)', 'сідло (7)']); a.legend(fontsize=8)
a.set_ylabel('E(центр на дефекті) − E(далеко)'); a.set_title('а) світу вигідно сидіти на додатній кривині,\nневигідно на від\'ємній', fontsize=10)
a = ax[1]; d = np.linspace(0, 2*np.pi, 200)
a.plot(np.degrees(d), np.sqrt(1 - d/(2*np.pi)), color='#2a6fdb'); a.plot([60], [np.sqrt(5/6)], 'o', color='#d9480f')
a.set_xlabel('вирізаний кут довкола світу δ, °'); a.set_ylabel('енергія межі / пласка'); a.set_xticks(range(0, 361, 60))
a.set_title('б) без ціни кривини вигода росте до δ = 360°:\nсвіт «згортався б» у замкнену кульку (гіпотеза)', fontsize=10)
plt.tight_layout(); plt.savefig('/mnt/project-files/figures/fig241_world_curvature.png', dpi=130); print(meas, pred)
