# Scripts as run (v1.4)

`rNNN/` holds the scripts of computation (NNN) of the project journal exactly as they were run, with their job lists
(`job*.sh`, `runall*.sh`, `jobs*.txt`). Comments are partly in Ukrainian and paths are absolute (`/mnt/project-files/model/s52/...`).
To rerun, replace that prefix with your output folder and point imports to `src/variant` (live model with the field sector:
`run268.py`; set `GNB=1` for the compiled gauge kernel `gauge_nb.py`, same physics to 1e-14, checked by `test_gauge_nb.py`)
or to `src/` (3D fabric and helpers: `fabgen.py`, `simNlib.py`, `local60.py`, `rnd51.py`, `tile48.py`, `birth48.py`).
The summaries these scripts produced are in `data/v14/rNNN`. Map to the paper (v1.4):

| Folder | Paper |
|---|---|
| r335 (also (337)) | Sec. 4.7, Fig. 9a,b: budget window of the variant with the field sector, L = 96 and 192, 8 seeds per budget; `fig337.py` |
| r348 | Sec. 4.7, Fig. 9c,d: pinned coarsening without field and elasticity, random vs lattice arrows; T = 1200 continuation (`job1200.sh`) |
| r346 | Sec. 4.7: coarsening with the field sector, with and without the elastic response |
| r353 | Sec. 4.7: scan of the arrow scale ℓ_τ; `lyap/`: twin runs (Lyapunov exponent 0.4–0.5) |
| r343, r344 | Sec. 5.1: a lattice disclination (cone) and light passing it (fronts at 58.2°) |
| r345, r347 | Sec. 5.1: a droplet on a cone and a saddle; candidate postulate deficit = 8πG·E (Regge lengths, lensing) |
| r352 | Sec. 5.1: link length from intensity, l ∝ (w₊²/|z|²)^p |
| r351 | Sec. 5.2: 3D test of the variant on Kuhn tetrahedra, L = 16, budgets 2.5–4.25 (`rq351f.py` = compiled force, checked by `test351f.py`) |

`paper/fig13_window_pinning.py` redraws Fig. 9 from `data/v14`.
