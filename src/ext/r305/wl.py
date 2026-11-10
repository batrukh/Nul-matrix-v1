import numpy as np, sys
z = np.load('/mnt/project-files/model/s52/r294/e0_ka1_kb2_a-1_L96.npz'); P, U = z['P'], z['U']; L = 96
cx, cy = 50.5, 25.11   # v1 (з r304)
# петля: шестикутник з радіусом R по вузлах трикутної сітки навколо v1; рухаємось кроками DIRS
D = {(1,0):(0,1), (0,1):(1,1), (-1,1):(2,1), (-1,0):(0,-1), (0,-1):(1,-1), (1,-1):(2,-1)}
def link(n, d):     # перенос із n+d у n
    m, s = D[d]
    if s == 1: return U[m][n[0] % L, n[1] % L]
    q = ((n[0] + d[0]) % L, (n[1] + d[1]) % L); return U[m][q].T
def loop(R):
    j0 = int(round(cy/(np.sqrt(3)/2))); i0 = int(round(cx - 0.5*j0))
    n = (i0 + R, j0); W = np.eye(5); path = []
    for d in [(-1,1),(-1,0),(0,-1),(1,-1),(1,0),(0,1)]:
        for _ in range(R): path.append((n, d)); n = (n[0] + d[0], n[1] + d[1])
    for n, d in path: W = W @ link(n, d)
    return W, path[0][0]
for R in [3, 6, 10, 14]:
    W, n0 = loop(R); Pn = P[n0[0] % L, n0[1] % L]
    ev, V = np.linalg.eigh(Pn); Wf = V.T @ W @ V
    print(R, 'tr', round(np.trace(W), 3), 'eig', np.round(np.linalg.eigvals(W), 3), '\n  W in Φ-frame diag', np.round(np.diag(Wf), 3), ' |[W,Φ]|', round(np.linalg.norm(W @ Pn - Pn @ W), 3))
