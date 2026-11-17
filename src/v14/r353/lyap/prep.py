"""(353) показник Ляпунова: дві копії з контрольної точки r348 (t=590), друга з збуренням z на 1e-8; далі до t=620 зі знімками щоодиниці."""
import pickle, numpy as np, os, sys
src = '/mnt/project-files/model/s52/r348/ck/%s/lc_s03.5_n100_L96_s1_tau16_iso_b1.3_ck.pkl'
for run in ('ge0.0025_L96_s1', 'ge0_L96_s1'):
    C = pickle.load(open(src % run, 'rb')); C['spl'] = []
    for tw, eps in (('A', 0.0), ('B', 1e-8)):
        D = dict(C); rng = np.random.default_rng(5)
        if eps: D['z'] = C['z'] + eps*(rng.standard_normal(C['z'].shape) + 1j*rng.standard_normal(C['z'].shape))
        d = f'{run}_{tw}'; os.makedirs(d + '/ck', exist_ok=True)
        pickle.dump(D, open(f'{d}/ck/' + os.path.basename(src % run), 'wb'))
