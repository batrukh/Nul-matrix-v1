# Director code (v1.3, Secs. 4.8–4.9)

Copy of `src/variant` with a live director n (n ≡ −n, a unit 5-vector per node) switched on by environment variables:
`DMU` (coupling μ in μ nᵀΦ²n, paper 0.03), `DK` (stiffness k in −Σ w (k/2)(nᵀUn′)², paper 0.09). Without them the code is `src/variant`.
`SNAP=1` adds domain bitmaps (ψ > 0.3) and positions of disclinations and Alice vortices to the snapshots; `DST` is the interval of the
defect log in field steps (20 = one time unit). Run through `runs/director.sh`. The defect log `*_dtrack.json` holds per entry
k (field step), aw (domain area fraction), nz2 (Alice vortex triangles), nd (director disclination triangles in domains).
