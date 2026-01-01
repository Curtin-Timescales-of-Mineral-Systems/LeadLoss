# 2025 peak picking manuscript bundle

This directory contains the data and Python scripts used to reproduce the **figures and tables** for the 2025 “peak picking” Pb-loss manuscript.

## Directory layout

- `data/inputs/` — input CSVs used by the workflows.
- `data/derived/` — derived datasets used by the manuscript scripts.
  - `ensemble_catalogue.csv` — CDC ensemble catalogue used by figures/tables.
  - `ks_diagnostics/` — per-run and bundled NPZ goodness surfaces (`*_runs_S.npz`) used for Figures 3, 5, and 8.
  - `reimink_discordance_dating/` — bootstrap/aggregate outputs used for DD figures and benchmark tables.
- `scripts/`
  - `scripts/tables/` — table reproduction scripts.
  - `scripts/figures/` — figure reproduction scripts.
  - `scripts/_util/` — shared helpers (path handling, benchmark definitions).
- `outputs/` — generated artefacts (tables/figures). These are included for convenience, and can be regenerated.
- `figures/` — manuscript-ready figure exports (included).

## Environment setup

From the **repository root**:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r papers/2025-peak-picking/requirements.txt
```

If you are running headless (e.g., on a server/CI runner), set a non-interactive matplotlib backend:

```bash
export MPLBACKEND=Agg
```

## Reproducing tables

All table scripts write into:

- `papers/2025-peak-picking/outputs/tables/`

Run (from repo root or any working directory):

```bash
python papers/2025-peak-picking/scripts/tables/tables_01_02_benchmark_definitions.py
python papers/2025-peak-picking/scripts/tables/tables03_to_08_benchmark_results.py
python papers/2025-peak-picking/scripts/tables/tables_09_concordant_fraction_sweep.py
python papers/2025-peak-picking/scripts/tables/tables_10_12_runtime_tables.py
```

## Reproducing figures

Most figure scripts write into:

- `papers/2025-peak-picking/outputs/figures/`

### Figure 1 — synthetic cases 1–4

```bash
# In headless mode use MPLBACKEND=Agg (recommended) or comment out plt.show().
python papers/2025-peak-picking/scripts/figures/fig01_synthetic_cases1to4.py
```

Expected outputs:

- `outputs/figures/cases1-4_Wetherill_grid.(svg|png|tiff)`
- `outputs/derived/synthetic/cases1to4_synth_*.csv`

### Figure 2 — synthetic cases 5–7

This script is CLI-driven. To keep output locations consistent with the other figure scripts, pass `--fig-dir`:

```bash
python papers/2025-peak-picking/scripts/figures/fig02_synthetic_cases5to7.py \
  --save-fig \
  --fig-dir papers/2025-peak-picking/outputs/figures \
  --formats svg,png,pdf
```

### Figures 3 and 5 — CDC goodness grids

These figures use NPZ goodness surfaces under `data/derived/ks_diagnostics/`.

```bash
python papers/2025-peak-picking/scripts/figures/fig03_fig05_cdc_goodness_grids.py \
  --ks-dir papers/2025-peak-picking/data/derived/ks_diagnostics \
  --no-show
```

### Figures 4 and 6 — DD likelihood grids

```bash
python papers/2025-peak-picking/scripts/figures/fig04_fig06_dd_likelihood_grids.py \
  --dd-dir papers/2025-peak-picking/data/derived/reimink_discordance_dating \
  --no-show
```

### Figure 8 — CDC upgrade diagnostic (default: sample 2A)

```bash
python papers/2025-peak-picking/scripts/figures/fig08_cdc_upgrade.py \
  --sample-id 2A \
  --no-show
```

## Notes

- If you intend to keep this repository lightweight, you can choose not to version-control `outputs/` and regenerate outputs locally. See the commented section in the repo root `.gitignore`.
