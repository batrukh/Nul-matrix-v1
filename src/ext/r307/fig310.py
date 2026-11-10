"""(310) рис. 205: однакова маса всім трьом G01, G02, G12"""
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
def low(f):
    for q in json.load(open(f))['modes']:
        if q['odd'] > 0.5 and q['q2'] > 0.5 and q['w2'] > 1e-4: return q['w']
MG = ['0.2', '0.3', '0.5']; x = np.arange(3); w = 0.26
fig, a = plt.subplots(figsize=(7, 4.3))
a.bar(x - w, [low(f'R20/m2_v_mg{m}.json') for m in MG], w, label='вихор, маса G02, G12 (307)')
a.bar(x, [low(f'R20/m2_v_all3_mg{m}.json') for m in MG], w, label='вихор, маса G01, G02, G12 (310)')
a.bar(x + w, [low(f'V20/m2_vac_rc0.0_all3_mg{m}.json') for m in MG], w, color='#999', label='без вихору, маса всім трьом')
for i, m in enumerate(MG): a.plot([i - 1.5*w, i + 1.5*w], [float(m)]*2, 'k:', lw=1)
a.set_xticks(x, [f'm = {m.replace(".", ",")}' for m in MG]); a.set_ylabel('найнижча заряджена непарна ω, r<20')
a.set_title('Однакова маса всій трійці SO(3): вихор тримає квант так само'); a.legend(fontsize=8); a.grid(alpha=.3, axis='y')
plt.tight_layout(); plt.savefig('fig205_all3_mass.png', dpi=130)
