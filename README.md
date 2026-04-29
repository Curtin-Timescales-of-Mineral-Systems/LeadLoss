![Curtin University: Timescales of Mineral Systems](resources/logo-linear.png)

# LeadLoss

LeadLoss is a Python-based tool for estimating the most likely timing of Pb-loss in discordant zircon samples. This repository includes:

- a cross-platform **GUI application** for interactive analysis, and
- the source code and tests for the current public code release.

This branch is a code-only public snapshot. Manuscript assets, figure bundles,
rerun outputs, and large derived datasets are distributed separately from this
repository.

## Download and installation

### Option 1: Standalone executables (recommended)

Standalone executables for Windows and macOS are provided as **GitHub Release assets**:

- https://github.com/Curtin-Timescales-of-Mineral-Systems/LeadLoss/releases

Download the appropriate file for your operating system and run it (no Python installation required).

### Option 2: Run the GUI from source (Python environment)

Clone the repository:

```bash
git clone https://github.com/Curtin-Timescales-of-Mineral-Systems/LeadLoss.git
cd LeadLoss
```

Create and activate a virtual environment, then install **GUI** dependencies:

```bash
python -m venv .venv-app
source .venv-app/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-app.txt
```

Run the GUI:

```bash
python src/application.py
```

Note: `soerp` is now optional in source installs. If it is unavailable on your platform,
the app falls back to deterministic math operations for the affected internals.

## Input requirements (GUI)

The GUI expects a CSV file where each row represents a single spot analysis. Required columns are:

- 238U/206Pb ratio
- 238U/206Pb uncertainty
- 207Pb/206Pb ratio
- 207Pb/206Pb uncertainty

During import, you can specify column names or indices (e.g., A, B, C, D or 1, 2, 3, 4). Optional columns (e.g., sample identifiers for batch processing) may also be included. Uncertainties may be specified as absolute values or percentages at the 1σ or 2σ confidence level.

## Outputs (GUI)

The GUI can export:

- optimal Pb-loss age estimates with empirical 2.5/97.5 percentile stability bounds
- K–S test statistics (p-values and D-values)
- individual Monte Carlo sampling results
- ensemble catalogue of Pb-loss age estimates with empirical 2.5/97.5 percentile stability bounds, age-mode metadata, and support values

## Troubleshooting

If a standalone executable does not run, confirm you downloaded the correct operating-system build from the Releases page.

If you run from source, confirm your environment is active and dependencies are installed from `requirements-app.txt`. If issues persist, please open a GitHub issue and include your operating system and the full error message/traceback.

## Citation

If you use LeadLoss in your research, please cite:

Mathieson, L. M., Kirkland, C. L., & Daggitt, M. L. (2025). Turning trash into treasure: Extracting meaning from discordant data via a dedicated application. *Geochemistry, Geophysics, Geosystems*, 26, e2024GC012066. https://doi.org/10.1029/2024GC012066
