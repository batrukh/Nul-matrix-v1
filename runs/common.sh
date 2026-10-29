H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; SRC="$H/../src"; OUT="$H/out"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
