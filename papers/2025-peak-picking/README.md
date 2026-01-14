# 2025 peak-picking manuscript bundle

This folder contains the data and Python scripts used to reproduce the **figures and tables** for the 2025 “peak-picking” Pb-loss manuscript.

## Directory layout

- `data/inputs/` — input CSVs used by the workflows.
- `data/derived/` — derived datasets used by the manuscript scripts.
  - `ensemble_catalogue.csv` — CDC ensemble catalogue used by figures/tables.
  - `ks_exports/` — exported K–S goodness summaries used in figures/tables.
  - `reimink_discordance_dating/` — bootstrap/aggregate outputs used for DD figures and benchmark tables.
  - `ks_diagnostics_npz.tar.gz` — archived per-run NPZ goodness surfaces required for Figures 3, 5, and 8 (see below).
- `scripts/`
  - `scripts/tables/` — table reproduction scripts.
  - `scripts/figures/` — figure reproduction scripts.
  - `scripts/_util/` — shared helpers (path handling, benchmark definitions).
- `outputs/` — generated artefacts (tables/figures). These can be regenerated.
- `figures/` — manuscript-ready figure exports (included for convenience).

## Environment setup

From the **repository root**:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r papers/2025-peak-picking/requirements.txt
```

If you are running headless (e.g., CI/server), set a non-interactive matplotlib backend:

```bash
export MPLBACKEND=Agg
```

## KS diagnostics NPZ bundle (required for Figures 3, 5, and 8)

To keep the Git repository lightweight, the per-run CDC goodness surfaces are provided as a single archive:

- `papers/2025-peak-picking/data/derived/ks_diagnostics_npz.tar.gz`

Extract it once (from the repository root):

```bash
tar -xzf papers/2025-peak-picking/data/derived/ks_diagnostics_npz.tar.gz   -C papers/2025-peak-picking/data/derived
```

This will create:

- `papers/2025-peak-picking/data/derived/ks_diagnostics/`

That extracted folder is intentionally ignored by git.

## One-command reproduction

From the repository root (or any working directory):

```bash
python papers/2025-peak-picking/scripts/run_all.py --clean
```

This regenerates all manuscript artefacts into:

- `papers/2025-peak-picking/outputs/tables/`
- `papers/2025-peak-picking/outputs/figures/`

Notes:

- `run_all.py` will extract `ks_diagnostics_npz.tar.gz` automatically if required.
- `outputs/` contains generated artefacts and can be deleted/regenerated.
- `figures/` contains manuscript-ready exports (included for convenience).

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

If a script opens interactive windows, add `--no-show` where supported, or run headless using `export MPLBACKEND=Agg`.

### Figure 1 — synthetic cases 1–4

```bash
python papers/2025-peak-picking/scripts/figures/fig01_synthetic_cases1to4.py
```

### Figure 2 — synthetic cases 5–7

This script is CLI-driven. To keep output locations consistent with the other figure scripts, pass `--fig-dir`:

```bash
python papers/2025-peak-picking/scripts/figures/fig02_synthetic_cases5to7.py   --save-fig   --fig-dir papers/2025-peak-picking/outputs/figures   --formats svg,png,pdf
```

### Figures 3 and 5 — CDC goodness grids

These figures use NPZ goodness surfaces under `data/derived/ks_diagnostics/` (created by extracting the NPZ bundle above).

```bash
python papers/2025-peak-picking/scripts/figures/fig03_fig05_cdc_goodness_grids.py   --ks-dir papers/2025-peak-picking/data/derived/ks_diagnostics   --no-show
```

### Figures 4 and 6 — DD likelihood grids

```bash
python papers/2025-peak-picking/scripts/figures/fig04_fig06_dd_likelihood_grids.py   --dd-dir papers/2025-peak-picking/data/derived/reimink_discordance_dating   --no-show
```

### Figure 7 — K–S goodness example (case 2A)

```bash
python papers/2025-peak-picking/scripts/figures/fig07_ks_goodness_case2a.py --no-show
```

### Figure 8 — CDC upgrade diagnostic (default: sample 2A)

This figure also requires the extracted `data/derived/ks_diagnostics/` folder.

```bash
python papers/2025-peak-picking/scripts/figures/fig08_cdc_upgrade.py   --sample-id 2A   --no-show
```

## Notes

- `outputs/` can be regenerated. If you prefer a lightweight clone, you can choose not to version-control `outputs/` (see the repo root `.gitignore`).
- If you are reviewing this repository and only want the manuscript-ready figures, see `papers/2025-peak-picking/figures/`.
