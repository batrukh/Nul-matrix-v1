"""rq383g (17.11, (396)): rq383 на прискореній силі fastpatch (та сама фізика; перевірка test396*.py). Аргументи й env — як у rq383."""
import os; os.environ["NB"] = "0"; os.environ.setdefault("DELTA", "1e-6")
import sys; sys.path[:0] = ["/mnt/project-files/model/s52/r396"]
import fastpatch; fastpatch.install()
exec(open("/mnt/project-files/model/s52/r383/rq383.py").read())
