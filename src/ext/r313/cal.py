import numpy as np, sys; from eff import *
L, R = int(sys.argv[1]), float(sys.argv[2])
n, U, r, _ = config(L, R, 3.0, 0.0)
for t, rc in [(0.1, 3.0), (0.1, 4.0), (0.2, 4.0)]:
    for h in [1.0, 1.2]:
        w, v = low(build(L, n, U, r, h=h, t=t, rc=rc), 8); we, _ = low(build(L, n, U, r, h=h, t=t, rc=rc, mode='even'), 4)
        print('t', t, 'rc', rc, 'h', h, 'odd ω', np.round(np.sqrt(np.abs(w)), 4), '| even', np.round(np.sqrt(np.abs(we)), 4), flush=True)
