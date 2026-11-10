"""(320) положення ядер: центроїд надлишку густини енергії (над вакуумом) у колах r < 5 біля ям. python3 core.py TAG"""
import numpy as np, sys
sys.path.insert(0, '/mnt/project-files/model/s52/r320'); from pair import *
def cores(P, U, n, R, phi, A, w):
    L = P.shape[0]; g = model(L, R, phi, A, w); e = g.edens(P, np.zeros_like(P), U, np.zeros_like(U)) + g.en(P, U, n)
    S = pos(L); out = []
    for v in verts(L, R, phi):
        r, D = wrapv(S - v, L); m = r < 5; ex = e[m] - np.median(e); ex = np.clip(ex, 0, None)
        out.append((v + (ex[:, None]*D[m]).sum(0)/ex.sum()).round(3).tolist() + [float(ex.sum())])
    return out
if __name__ == '__main__':
    z = np.load(f'/mnt/project-files/model/s52/r320/fin/{sys.argv[1]}.npz')
    print('ями', [list(np.round(v, 3)) for v in verts(z['P'].shape[0], float(z['R']), float(z['phi']))])
    print('ядра', cores(z['P'], z['U'], z['n'], float(z['R']), float(z['phi']), float(z['A']), float(z['w'])), 'it', int(z['it']), 'fm', float(z['fm']))
