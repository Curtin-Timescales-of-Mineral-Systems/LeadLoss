# 2025 peak-picking manuscript bundle

This folder contains the data and Python scripts used to reproduce the **figures and tables** for the 2025 “peak-picking” Pb-loss manuscript.

## Directory layout

- `data/inputs/` — input CSVs used by the workflows.
  - `data/inputs/ga_gawler_fig08/` — analysis-ready inputs for the illustrative natural dataset used in Fig. 08 (Gawler Craton). See the README in that folder for dataset links, filters, and column definitions.
- `data/derived/` — derived datasets used by the manuscript scripts.
  - `ensemble_catalogue.csv` — CDC ensemble catalogue used by figures/tables (includes the peak ages/CIs reported for the Gawler example).
  - `ks_exports/` — exported K–S goodness summaries used in figures/tables.
  - `reimink_discordance_dating/` — bootstrap/aggregate outputs used for DD figures and benchmark tables.
  - `ks_diagnostics_npz.tar.gz` — archived per-run NPZ goodness surfaces required for Figs. 3, 5, and 9 (extracts to `data/derived/ks_diagnostics/`).
  - `ks_diagnostics_gawler_npz.tar.gz` — archived NPZ diagnostics required for Fig. 08 (extracts to `data/derived/ks_diagnostics_gawler/`).
- `scripts/`
  - `scripts/tables/` — table reproduction scripts.
  - `scripts/figures/` — figure reproduction scripts.
  - `scripts/_util/` — shared helpers (path handling, benchmark definitions).
- `outputs/` — generated artefacts (tables/figures). These can be regenerated and are not required to be version-controlled.
- `figures/` — manuscript-ready figure exports.

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

## NPZ diagnostics bundles

To keep the Git repository lightweight, per-run CDC goodness surfaces are provided as tarballs. The extracted folders are intentionally ignored by git.

### Benchmark NPZ bundle (Figs. 3, 5, 9)

- `papers/2025-peak-picking/data/derived/ks_diagnostics_npz.tar.gz`

Extract once (from the repository root):

```bash
tar -xzf papers/2025-peak-picking/data/derived/ks_diagnostics_npz.tar.gz \
  -C papers/2025-peak-picking/data/derived
```

This creates:

- `papers/2025-peak-picking/data/derived/ks_diagnostics/`

### Gawler NPZ bundle (Fig. 08)

- `papers/2025-peak-picking/data/derived/ks_diagnostics_gawler_npz.tar.gz`

Extract once (from the repository root):

```bash
tar -xzf papers/2025-peak-picking/data/derived/ks_diagnostics_gawler_npz.tar.gz \
  -C papers/2025-peak-picking/data/derived
```

This creates:

- `papers/2025-peak-picking/data/derived/ks_diagnostics_gawler/`

## One-command reproduction

From the repository root (or any working directory):

```bash
python papers/2025-peak-picking/scripts/run_all.py --clean
```

This regenerates manuscript artefacts into:

- `papers/2025-peak-picking/outputs/tables/`
- `papers/2025-peak-picking/outputs/figures/`

Notes:

- `run_all.py` runs figure scripts headlessly (MPLBACKEND=Agg).
- `run_all.py` will extract required NPZ tarballs automatically if needed.
- `outputs/` contains generated artefacts and can be deleted/regenerated.

## Fresh Ensemble V2 Anchor-Clustered Bundles

Fresh anchor-clustered reruns are written into settings-named folders under:

- `papers/2025-peak-picking/data/derived/ensemble_v2_anchor_clustered_*`

Those bundles are separate from the legacy manuscript `data/derived/` folder on purpose, so they do not overwrite the originally submitted artefacts.

To build a fresh clustering bundle, use:

```bash
python papers/2025-peak-picking/scripts/regenerate_cdc_clustering_bundle.py
```

To regenerate the CDC-derived benchmark figures/tables from one of those bundles into a separate output folder, use:

```bash
python papers/2025-peak-picking/scripts/run_clustering_bundle.py \
  --derived-root papers/2025-peak-picking/data/derived/ensemble_v2_anchor_clustered_sigma1_1to2000_nodes100_mc100 \
  --clean
```

Run that from the paper-generation Python environment, i.e. an interpreter with
`papers/2025-peak-picking/requirements.txt` installed. The GUI app interpreter
used for `src/application.py` is not necessarily the same environment.

This writes to:

- `papers/2025-peak-picking/outputs/ensemble_v2_anchor_clustered/<bundle_name>/tables/`
- `papers/2025-peak-picking/outputs/ensemble_v2_anchor_clustered/<bundle_name>/figures/`

Notes:

- `run_clustering_bundle.py` regenerates the CDC-derived benchmark artefacts only.
- It does not touch the legacy `outputs/tables/` or `outputs/figures/` folders unless you point it there explicitly.
- It reuses the existing DD comparison directory (`data/derived/reimink_discordance_dating/`) for tables that still compare against DD.
- The static synthetic illustration figures (Figs. 1–2) and the Gawler natural-example figure are not part of this clustering-bundle runner.

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

```bash
python papers/2025-peak-picking/scripts/figures/fig02_synthetic_cases5to7.py \
  --save-fig \
  --fig-dir papers/2025-peak-picking/outputs/figures \
  --formats svg,png,pdf
```

### Figures 3 and 5 — CDC goodness grids

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

### Figure 7 — K–S goodness example (case 2A)

```bash
python papers/2025-peak-picking/scripts/figures/fig07_ks_goodness_case2a.py \
  --no-show
```

### Figure 8 — illustrative natural example (Gawler Craton)

Inputs:
- `papers/2025-peak-picking/data/inputs/ga_gawler_fig08/` (see folder README)
- `papers/2025-peak-picking/data/derived/ks_diagnostics_gawler/` (created by extracting `ks_diagnostics_gawler_npz.tar.gz`)

Figure script:
- `papers/2025-peak-picking/scripts/figures/fig08_gawler_natural_example.py`

Run:

```bash
python papers/2025-peak-picking/scripts/figures/fig08_gawler_natural_example.py \
  --no-show \
  --fig-dir papers/2025-peak-picking/outputs/figures \
  --outfile f08
```

### Figure 9 — CDC upgrade diagnostic (default: sample 2A)

This figure requires the extracted `data/derived/ks_diagnostics/` folder (created by extracting `ks_diagnostics_npz.tar.gz`).

```bash
python papers/2025-peak-picking/scripts/figures/fig09_cdc_upgrade.py \
  --sample-id 2A \
  --no-show \
  --fig-dir papers/2025-peak-picking/outputs/figures
```

## Notes

- If you are reviewing this repository and only want the manuscript-ready figures, see `papers/2025-peak-picking/figures/`.
