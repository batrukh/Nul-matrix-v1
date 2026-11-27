"""rq447c (21.11): rq447 з швидким K (kcheb, розклад Чебишова). Ті самі аргументи; решта без змін."""
import sys; sys.path[:0] = ["/mnt/project-files/model/s52/r447"]
import rq447 as W, kcheb
_b = W.build
def build(*a, **k):
    s, f, mod, QV, tqn, bad, tag = _b(*a, **k); s.K = kcheb.make_K(s.P, s.n); print("kcheb", s.K.info, flush=True); return s, f, mod, QV, tqn, bad, tag
W.build = build
if __name__ == "__main__":
    kw = {k: (v if k == "lat" else (int(v) if k in ("seed", "check") else float(v))) for k, v in (a.split("=") for a in sys.argv[1:])}
    print(W.run(**kw))
