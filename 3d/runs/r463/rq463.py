"""rq463 (22.11, (463)): rq383g (sgn=0, прискорена сила) на сітці Куна вимірності D (env D, за замовч. 4). Решта аргументів як у rq383."""
import os, sys
os.environ["NB"] = "0"; os.environ.setdefault("DELTA", "1e-6")
sys.path[:0] = ["/mnt/project-files/model/s52/r396", "/mnt/project-files/model/s52/r351", "/mnt/project-files/model/s52", "/mnt/project-files/model/s52/r463"]
import fastpatch; fastpatch.install()
import kund; kund.install(int(os.environ.get("D", "4")))
exec(open("/mnt/project-files/model/s52/r383/rq383.py").read())
