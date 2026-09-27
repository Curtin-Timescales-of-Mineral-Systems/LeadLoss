# LeadLoss 2.1.0-rc1 release notes

This is a release candidate based directly on the official LeadLoss v2.0.3
tag. It is intended for final testing before a v2.1.0 release.

## What changed

- Native Wetherill CSV input is supported using 207Pb/235U and 206Pb/238U
  ratios, their uncertainties, and an optional error-correlation (rho) column.
- An optional rho column can also be supplied with Tera-Wasserburg input,
  allowing exact paired coordinate-system test files to retain covariance.
- Concordia display and CDC calculation spaces can be selected independently
  as Tera-Wasserburg, Wetherill, or the same space as the imported data.
- Monte Carlo values are drawn once in the native input space. Correlated
  Wetherill uncertainty is included when rho is supplied, and the realised
  draws are transformed when another calculation space is selected.
- Summary and per-sample concordia plots use the selected geometry and show
  covariance-aware uncertainty ellipses instead of independent error crosses.
- Output tables and CSV exports record the CDC calculation space.
- Ensemble results use the plain labels "Ensemble peak" and
  "Boundary-limited".
- Direct support and winner support are calculated and labelled separately,
  consistent with the published ensemble-method terminology.
- The interface has a restrained visual refresh with no intended modelling
  changes.
- GitHub release builds now require the test suite to pass first, and tests run
  again on each target build platform.
- The macOS build uses a repository-owned multi-resolution app icon instead of
  relying on a PyInstaller installation's optional default icon.
- The packaged macOS app now records the 2.1.0 marketing version and a stable
  Curtin bundle identifier instead of PyInstaller's `0.0.0` default.
- The release workflow produces separate native Intel and Apple-Silicon macOS
  archives rather than requiring Apple-Silicon users to rely on Rosetta.

## Deliberate scope decision

Kolmogorov-Smirnov remains the only dissimilarity measure in this release
candidate. Other measures are not included because their score scaling and
ensemble thresholds have not been calibrated.

## Checks completed locally

- Full automated unit/regression suite.
- Four input/calculation combinations: TW→TW, TW→Wetherill,
  Wetherill→Wetherill and Wetherill→TW.
- Covariance transformation and round-trip tests.
- Wetherill concordia and discordia-intersection tests.
- Preservation of the historical TW random-draw order for uncorrelated input.
- TW rho input and covariance-ellipse geometry tests.
- Off-screen construction of the themed main window and native-Wetherill
  import dialog.
- Packaged-GUI imports of a real Tera-Wasserburg CSV and a real native
  Wetherill CSV with rho, including inspection of the tables, concordia plots
  and exported peak catalogues.
- Frozen real-data comparison against v2.0.3 using the historical
  uncorrelated Tera-Wasserburg path. Concordance classifications, optimal
  ages, goodness-of-fit curves and ensemble catalogues were unchanged.

## Final release gates

Before tagging v2.1.0:

1. Run the GitHub workflow and confirm the Windows, macOS Intel and macOS
   Apple-Silicon tests/builds.
2. Manually import at least one real TW CSV and one real native Wetherill CSV
   with rho, and inspect tables, concordia plots and exported CSV files.
3. Compare a frozen TW benchmark against v2.0.3 and record any expected change.
4. Replace the release-candidate version string with `2.1.0`.
5. Update `CITATION.cff` and the README citation only after the new archive DOI
   and release date are known.
