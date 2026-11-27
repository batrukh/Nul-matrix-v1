"""світи стану local60: компоненти ψ>0 (6 сусідів), найглибший вузол кожного (BFS від краю)"""
import numpy as np, sys
from collections import deque
from phi90 import NB, coords
def worlds(psi):
    L = psi.shape[0]; W = psi > 0; lab = -np.ones((L, L), int); comps = []
    for x in range(L):
        for y in range(L):
            if W[x, y] and lab[x, y] < 0:
                q = deque([(x, y)]); lab[x, y] = len(comps); c = []
                while q:
                    u = q.popleft(); c.append(u)
                    for dx, dy in NB:
                        v = ((u[0]+dx) % L, (u[1]+dy) % L)
                        if W[v] and lab[v] < 0: lab[v] = len(comps); q.append(v)
                comps.append(c)
    depth = np.where(W, 10**6, 0); q = deque([(x, y) for x in range(L) for y in range(L) if not W[x, y]])
    while q:
        u = q.popleft()
        for dx, dy in NB:
            v = ((u[0]+dx) % L, (u[1]+dy) % L)
            if depth[v] > depth[u] + 1: depth[v] = depth[u] + 1; q.append(v)
    out = []
    for i, c in enumerate(sorted(comps, key=len, reverse=True)):
        c = np.array(c); dd = depth[c[:, 0], c[:, 1]]; k = np.argmax(dd)
        out.append(dict(n=len(c), deep=tuple(c[k]), depth=int(dd[k]), nodes=c))
    return out, lab, depth
if __name__ == "__main__":
    psi = np.load(sys.argv[1])["psi"]; L = int(round(np.sqrt(len(psi)))); ws, _, _ = worlds(psi.reshape(L, L))
    X, Y = coords(L)
    for w in ws[:8]: print(w["n"], "deep", w["deep"], "depth", w["depth"], "xy %.1f,%.1f" % (X[w["deep"]], Y[w["deep"]]))
