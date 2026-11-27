"""fig302: (450) світи на ґратках, вирощених з поля (R, 448): частка світу на вузлах ОЦК і на вузлах-дефектах; частка дефектів ґратки."""
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
r = json.load(open('ana450.json')); k = np.array([q['seed'] for q in r])
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.6), dpi=150)
a.bar(k-0.2, [q['world_on_bcc'] for q in r], 0.4, label='світ на вузлах ОЦК')
a.bar(k+0.2, [np.nan if q['world_on_defects'] is None or q['defect_lat']==0 else q['world_on_defects'] for q in r], 0.4, label='світ на вузлах-дефектах')
a.set_xlabel('ґратка R (зерно)'); a.set_ylabel('частка вузлів у світі'); a.set_ylim(0, 1.15); a.set_xticks(k); a.legend(fontsize=8, ncol=2, loc='upper center')
a.set_title('а) світ 8/8, 85–87 % ґратки')
b.bar(k, [100*q['defect_lat'] for q in r], color='0.5'); b.set_xticks(k); b.set_xlabel('ґратка R (зерно)'); b.set_ylabel('дефекти ґратки (CNA не-ОЦК), %')
b.set_title('б) частка дефектів (з2 — ідеальна)')
fig.suptitle('(450) Світ народжується на ґратці, вирощеній з поля'); fig.tight_layout(); fig.savefig('/mnt/project-files/figures/fig302_3d_svit_na_gratci_z_polya.png')
