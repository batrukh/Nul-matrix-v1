"""(317) А: нитки обирають сусідів. Комбінаторна тріангуляція тору (позиції не потрібні), сусідство змінюється перемиканням
ребра 2→2 (Пахнер). Бюджет вузла (місцеве збереження інтенсивності) ділиться порівну між його k нитками: нитка u–v має
інтенсивність I = (1/k_u + 1/k_v)/2 (MODE avg) або I = 1/sqrt(k_u k_v) (MODE geo). Енергія комірок (трійка ниток):
E = −Σ_трикутники Π sqrt(I_e) (амплітуда комірки = добуток амплітуд ниток). MODE zero — контроль без енергії.
lam — додатковий член кривини λΣ(k−6)² (вкладений руками, для порівняння). Метрополіс з охолодженням T0 → T1. Старт — Делоне з випадкових точок (скло)."""
import sys, json, numpy as np, time
from scipy.spatial import Delaunay
def delaunay_torus(L, jit, seed):
    rng = np.random.default_rng(seed); n = L*L
    i, j = np.divmod(np.arange(n), L); P = np.c_[(i + j/2) % L, j*1.0]
    P = (P + jit*rng.uniform(-0.5, 0.5, (n, 2))) % L if jit < 1 else rng.uniform(0, L, (n, 2))
    img = np.concatenate([P + L*np.array([dx, dy]) for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
    tri = Delaunay(img).simplices; c = img[tri].mean(1)
    keep = (c[:, 0] >= 0) & (c[:, 0] < L) & (c[:, 1] >= 0) & (c[:, 1] < L)
    return P, [tuple(sorted(t)) for t in (tri[keep] % n)]
class Tri:
    def __init__(s, n, faces, mode):
        s.n = n; s.mode = mode; s.adj = [set() for _ in range(n)]; s.opp = {}; s.faces = set()
        for f in faces: s.add_face(f)
        s.k = np.array([len(a) for a in s.adj])
    def add_face(s, f):
        a, b, c = f; s.faces.add(tuple(sorted(f)))
        for u, v, w in ((a, b, c), (b, c, a), (a, c, b)):
            e = (min(u, v), max(u, v)); s.opp.setdefault(e, set()).add(w); s.adj[u].add(v); s.adj[v].add(u)
    def I(s, u, v):
        ku, kv = s.k[u], s.k[v]
        return (1/ku + 1/kv)/2 if s.mode.startswith("avg") else 1/np.sqrt(ku*kv)
    def fE(s, f):
        if s.mode == "zero": return 0.0
        a, b, c = f; return (1 if s.mode.endswith("+") else -1)*np.sqrt(s.I(a, b)*s.I(b, c)*s.I(a, c))   # «+»: комірка коштує енергію (знак обернено)
    def faces_at(s, nodes):
        out = set()
        for u in nodes:
            for v in s.adj[u]:
                for w in s.opp[(min(u, v), max(u, v))]: out.add(tuple(sorted((u, v, w))))
        return out
    def energy(s): return sum(s.fE(f) for f in s.faces) + s.lam*float(((s.k-6)**2).sum())
    def try_flip(s, u, v, T, rng):
        e = (min(u, v), max(u, v)); o = s.opp.get(e)
        if o is None or len(o) != 2: return False
        a, b = tuple(o)
        if b in s.adj[a] or s.k[u] <= 3 or s.k[v] <= 3: return False
        nodes = {u, v, a, b}; F0 = s.faces_at(nodes); E0 = sum(s.fE(f) for f in F0) + s.lam*sum((s.k[x]-6)**2 for x in nodes)
        s.flip(u, v, a, b); F1 = s.faces_at(nodes); E1 = sum(s.fE(f) for f in F1) + s.lam*sum((s.k[x]-6)**2 for x in nodes); dE = E1 - E0
        if dE <= 0 or (T > 0 and rng.random() < np.exp(-dE/T)): return dE
        s.flip(a, b, u, v); return False
    def flip(s, u, v, a, b):   # ребро u–v (комірки u v a, u v b) → a–b (комірки a b u, a b v)
        for f in ((u, v, a), (u, v, b)): s.del_face(f)
        s.adj[u].discard(v); s.adj[v].discard(u); s.opp.pop((min(u, v), max(u, v)), None)
        s.add_face((a, b, u)); s.add_face((a, b, v))
        s.k[u] -= 1; s.k[v] -= 1; s.k[a] += 1; s.k[b] += 1
    def del_face(s, f):
        a, b, c = f; s.faces.discard(tuple(sorted(f)))
        for x, y, w in ((a, b, c), (b, c, a), (a, c, b)): s.opp[(min(x, y), max(x, y))].discard(w)
def grains(s):
    """Зерна: зв'язні кластери вузлів, у яких сам вузол і всі сусіди мають 6 сусідів."""
    good = np.array([s.k[u] == 6 and all(s.k[v] == 6 for v in s.adj[u]) for u in range(s.n)])
    seen = np.zeros(s.n, bool); sizes = []
    for u in np.where(good)[0]:
        if seen[u]: continue
        st = [u]; seen[u] = True; c = 0
        while st:
            x = st.pop(); c += 1
            for y in s.adj[x]:
                if good[y] and not seen[y]: seen[y] = True; st.append(y)
        sizes.append(c)
    return sorted(sizes, reverse=True)
def stats(s, t):
    g = grains(s); return dict(t=t, E=s.energy()/s.n, f6=float((s.k == 6).mean()), kstd=float(s.k.std()), kmin=int(s.k.min()), kmax=int(s.k.max()),
                               g1=g[0] if g else 0, g2=g[1] if len(g) > 1 else 0, ng=len(g), fgood=sum(g)/s.n)
if __name__ == "__main__":
    A = dict(L=50, jit=1.0, seed=1, mode="avg", lam=0.0, sweeps=400, T0=0.003, T1=1e-5, out="run")
    for a in sys.argv[1:]: k, v = a.split("="); A[k] = type(A[k])(v)
    P, F = delaunay_torus(A["L"], A["jit"], A["seed"]); s = Tri(A["L"]**2, F, A["mode"]); s.lam = A["lam"]; rng = np.random.default_rng(A["seed"] + 100)
    assert len(s.faces) == 2*s.n and len(s.opp) == 3*s.n
    log = [stats(s, 0)]; print(log[-1], flush=True); t0 = time.time()
    for sw in range(1, A["sweeps"] + 1):
        T = A["T0"]*(A["T1"]/A["T0"])**(sw/A["sweeps"]); edges = list(s.opp.keys()); acc = 0
        for idx in rng.permutation(len(edges)):
            u, v = edges[idx]
            if (min(u, v), max(u, v)) in s.opp and s.try_flip(u, v, T, rng) is not False: acc += 1
        if sw % 20 == 0 or sw == A["sweeps"]:
            st = stats(s, sw); st.update(T=T, acc=acc/len(edges), sec=round(time.time()-t0)); log.append(st); print(st, flush=True)
    json.dump(dict(args=A, log=log, k=s.k.tolist(), edges=[[int(a), int(b)] for a, b in s.opp], pos=P.tolist()), open(A["out"] + ".json", "w"))
