"""(304) спектр ферміона на соті поверх вихору бази. python3 run304.py MODE G  (MODE vort|vac)"""
import numpy as np, sys, json, time
sys.path.insert(0, '.'); sys.path.insert(0, '/mnt/project-files/model/s52/r268/code')
import hc, phi90 as PH
from scipy.sparse.linalg import eigsh, splu, LinearOperator
MODE, G = sys.argv[1], float(sys.argv[2]); K = int(sys.argv[3]) if len(sys.argv) > 3 else 80
z = np.load('/mnt/project-files/model/s52/r294/e0_ka1_kb2_a-1_L96.npz'); L = 96
P0 = np.diag(np.sort(np.linalg.eigvalsh(PH.vacuum(-1.0, 0.1))))
if MODE == 'vort': P, U = z['P'], z['U']
elif MODE == 'core':                                                     # контроль: те саме ядро (власні значення Φ), але без скруту рамки й голономії
    P = np.zeros((L, L, 5, 5)); P[..., range(5), range(5)] = np.linalg.eigvalsh(z['P']); U = np.broadcast_to(np.eye(5), (3, L, L, 5, 5)).copy()
else: P = np.broadcast_to(P0, (L, L, 5, 5)).copy(); U = np.broadcast_to(np.eye(5), (3, L, L, 5, 5)).copy()
ev = np.linalg.eigvalsh(z['P']); dfc = np.abs(ev - np.diag(P0)).sum(-1)          # калібрувально інваріантний «дефект»
X, Y = PH.coords(L); i1 = np.argmax(dfc); m2 = PH.tdist(X, Y, X.flat[i1], Y.flat[i1], L) > 15; i2 = np.argmax(np.where(m2, dfc, 0))
cen = [(float(X.flat[i1]), float(Y.flat[i1]), float(dfc.flat[i1])), (float(X.flat[i2]), float(Y.flat[i2]), float(dfc.flat[i2]))]
t0 = time.time(); H = hc.build(P, U, g=G)
lu = splu(H.tocsc()); Op = LinearOperator(H.shape, matvec=lambda x: lu.solve(x), dtype=float)
w, v = eigsh(Op, k=K, which='LM', ncv=3*K, tol=1e-9); w = 1/w; o = np.argsort(w); w, v = w[o], v[:, o]
XA, YA = hc.coordsAB(L); N = L*L
dens = (np.abs(v)**2).reshape(2, N, 5, K).sum(2).reshape(2*N, K)               # густина на трикутниках
r1 = PH.tdist(XA, YA, cen[0][0], cen[0][1], L); r2 = PH.tdist(XA, YA, cen[1][0], cen[1][1], L)
w1 = dens[r1 < 8].sum(0); w2 = dens[r2 < 8].sum(0)
fl = (np.abs(v)**2).reshape(2, N, 5, K).sum((0, 1))                              # вага по ароматах (у рамці вузла)
rr = np.sqrt((dens*np.minimum(r1, r2)[:, None]**2).sum(0))
gap = G*abs(np.diag(P0)).min()
res = dict(mode=MODE, g=G, gap=gap, E=w.tolist(), w1=w1.tolist(), w2=w2.tolist(), rrms=rr.tolist(), flav=fl.T.tolist(), cen=cen, sec=time.time() - t0)
json.dump(res, open(f'r304_{MODE}_g{G}.json', 'w'))
np.savez_compressed(f'r304_{MODE}_g{G}_dens.npz', dens=dens[:, np.abs(w) < gap*1.02].astype(np.float32), E=w[np.abs(w) < gap*1.02], cen=np.array(cen))
ing = np.abs(w) < gap - 1e-3
print('centres', cen, 'gap', gap, 'in-gap', ing.sum(), 'sec', round(time.time() - t0))
for e_, a, b_, r_ in zip(w[ing], w1[ing], w2[ing], rr[ing]): print(f'{e_:+.4f}  v1 {a:.2f}  v2 {b_:.2f}  r {r_:.1f}')
