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

The GUI accepts either Tera-Wasserburg or native Wetherill isotope ratios. Each CSV row represents one spot analysis.

For Tera-Wasserburg input, the required columns are:

- 238U/206Pb ratio
- 238U/206Pb uncertainty
- 207Pb/206Pb ratio
- 207Pb/206Pb uncertainty
- error correlation (rho), if available

For native Wetherill input, the required columns are:

- 207Pb/235U ratio
- 207Pb/235U uncertainty
- 206Pb/238U ratio
- 206Pb/238U uncertainty
- error correlation (rho), if available

During import, choose the input ratio system and specify column names or indices (e.g., A, B, C, D or 1, 2, 3, 4). Optional columns, such as sample identifiers for batch processing, may also be included. Uncertainties may be absolute or percentages at 1σ or 2σ. The rho field is optional in either coordinate system. If it is blank, the app assumes that the two reported ratio uncertainties are independent; native Wetherill imports show an explicit warning because correlation is commonly supplied for those ratios.

The concordia display and CDC calculation space can be set independently to Tera-Wasserburg, Wetherill, or “same as input”. Monte Carlo values are sampled once in the native input space, including rho when available, and the same realised values are transformed when another calculation space is selected. Concordia plots show covariance-aware analytical uncertainty ellipses rather than independent horizontal and vertical error bars.

## Outputs (GUI)

The GUI can export:

- optimal Pb-loss age estimates with empirical 2.5/97.5 percentile stability bounds
- K–S test statistics (p-values and D-values)
- individual Monte Carlo sampling results
- ensemble catalogue of Pb-loss age estimates with empirical 2.5/97.5 percentile stability bounds, evidence class, and two separately calculated support values
- per-sample goodness-of-fit curve CSV files for custom plotting
- per-sample heatmap density CSV files for custom plotting

### Interpreting ensemble results

- **Ensemble peak:** passed the full ensemble peak and support rules.
- **Boundary-limited:** the result is concentrated against the young end of the tested search window, so it is reported as a one-sided boundary mode rather than a resolved interior peak.

**Direct support** is the percentage of Monte Carlo runs containing an accepted per-run peak inside the reported stability window. **Winner support** is the percentage of runs in which the peak assigned to that window is the run's preferred solution. A recurring secondary peak can therefore have high direct support but lower winner support. These measures answer different questions; neither is a confidence level.

## Troubleshooting

If a standalone executable does not run, confirm you downloaded the correct operating-system build from the Releases page.

If you run from source, confirm your environment is active and dependencies are installed from `requirements-app.txt`. If issues persist, please open a GitHub issue and include your operating system and the full error message/traceback.

## Citation

If you use LeadLoss in your research, please cite the software release:

Mathieson, L. M., & Daggitt, M. (2026). *LeadLoss* (Version 2.1.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.14039112

If you use the LeadLoss method, please also cite:

Mathieson, L. M., Kirkland, C. L., & Daggitt, M. L. (2025). Turning trash into treasure: Extracting meaning from discordant data via a dedicated application. *Geochemistry, Geophysics, Geosystems*, 26, e2024GC012066. https://doi.org/10.1029/2024GC012066
