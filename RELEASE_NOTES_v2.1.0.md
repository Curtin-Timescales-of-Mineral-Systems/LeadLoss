# LeadLoss 2.1.0 release notes

LeadLoss 2.1.0 adds native Wetherill input and modelling alongside Tera-Wasserburg.

## What changed

- Native Wetherill CSV input is supported using 207Pb/235U and 206Pb/238U
  ratios, their uncertainties, and an optional error-correlation (rho) column.
- An optional rho column can also be supplied with Tera-Wasserburg input.
- Concordia display and CDC calculation spaces can be selected independently
  as Tera-Wasserburg, Wetherill, or the same space as the imported data.
- Monte Carlo values are drawn once in the native input space. Correlated
  uncertainty is included when rho is supplied, and the same realised values
  are transformed when another calculation space is selected.
- If rho is left blank, zero correlation is assumed.
- Summary and per-sample concordia plots use the selected geometry and show
  covariance-aware uncertainty ellipses.
- Output tables and CSV exports record the CDC calculation space.
- Ensemble results use the labels "Ensemble peak" and "Boundary-limited".
  The latter indicates that the preferred age lies at the young end of the
  modelled range and cannot be resolved as an interior peak.
- Direct support and winner support are calculated and labelled separately.
- The interface has been updated without changing the intended CDC workflow.
- Release builds run the test suite on Linux, Windows, Intel macOS and
  Apple-silicon macOS before the platform archives are produced.

## Scope

Kolmogorov-Smirnov remains the dissimilarity measure used by LeadLoss. Other
measures are not included because their score scaling and ensemble thresholds
have not been calibrated.

## Validation

Local validation included:

- the full automated unit and regression suite;
- all four input/calculation combinations: TW to TW, TW to Wetherill,
  Wetherill to Wetherill and Wetherill to TW;
- covariance transformation and round-trip tests;
- Wetherill concordia and discordia-intersection tests;
- preservation of the historical TW random-draw order for uncorrelated input;
- TW rho input and covariance-ellipse tests;
- construction of the main window and native-Wetherill import dialog;
- packaged-GUI imports of real Tera-Wasserburg and native Wetherill files; and
- comparison with LeadLoss 2.0.3 using the historical uncorrelated
  Tera-Wasserburg path.

In the frozen Tera-Wasserburg comparison, concordance classifications, optimal
ages, goodness-of-fit curves and ensemble catalogues were unchanged from
LeadLoss 2.0.3.
