"""rq388g (17.11, (396)): rq388 (продовження з готового стану) на прискореній силі fastpatch."""
import os; os.environ["NB"] = "0"; os.environ.setdefault("DELTA", "1e-6")
import sys; sys.path[:0] = ["/mnt/project-files/model/s52/r396"]
import fastpatch; fastpatch.install()
exec(open("/mnt/project-files/model/s52/r388/rq388.py").read())
