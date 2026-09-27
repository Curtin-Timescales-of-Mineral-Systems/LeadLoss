# LeadLoss 2.1.0-rc1 release-gate report

Date: 27 September 2026

Commit initially tested: `7efe74b8`

No release or version tag was created during these checks.

## Packaged macOS application

The release candidate was built from `packaging/LeadLoss-macos.spec` and run as
a packaged `.app`, rather than from the Python source tree.

## Real Tera-Wasserburg import

Input file: `GA_C4_three_samples_paired_TW.csv`

SHA-256: `4b70bdcd8df948382f5bace5143333c5c8d9fabf6fa0f5b36c285b408a8e3446`

Import settings verified in the dialog and in the imported table:

- Tera-Wasserburg ratios: 238U/206Pb and 207Pb/206Pb;
- percentage 1-sigma uncertainties for both ratios;
- rho read from column F;
- concordia display in the input coordinate system.

The first attempted GUI import was discarded because programmatic activation
of the uncertainty radio buttons did not change the dialog's internal state.
The file was re-imported with the settings visibly confirmed before processing.
That discarded attempt is not used as validation evidence.

With 2-sigma ellipse classification, a 1--2000 Ma grid of 200 nodes and 50
Monte Carlo runs, sample `78206019` contained 9 concordant and 19 normally
discordant analyses. The packaged GUI returned an ensemble peak at 292.96 Ma
(224.26--352.52 Ma stability interval) in Tera-Wasserburg space. The exported
catalogue is `real_data_gui/TW_78206019_ensemble.csv`.

## Real native Wetherill import

Input file: `GA_R2017715_C4_native_wetherill.csv`

SHA-256: `450f689460d39c6ae8240b04cc9b5efac593de41f4a1e760c9378f2efa7eef2d`

Import settings verified in the dialog and in the imported table:

- Wetherill ratios: 207Pb/235U and 206Pb/238U;
- percentage 1-sigma uncertainties for both ratios;
- rho read from column F;
- concordia display and calculation in Wetherill space.

The imported table contained plausible ratios, percentage uncertainties and
rho values, and the concordia plot displayed covariance ellipses of sensible
size and orientation. With the same calculation settings as above, the sample
contained 47 concordant and 47 normally discordant analyses. The optimal age
was 171.77 Ma. The ensemble curve had a broad best-fit age at 171.77 Ma
(103.71--232.04 Ma stability interval). The exported catalogue is
`real_data_gui/Wetherill_GA_R2017715_ensemble.csv`.

## Frozen historical TW comparison with v2.0.3

The real paired TW file was rerun without rho so that both versions used the
historical uncorrelated TW path. Both versions used percentage 1-sigma input
errors, 2-sigma ellipse classification, a 1--2000 Ma grid of 200 nodes, and 50
Monte Carlo runs.

For all three samples, v2.0.3 and v2.1.0-rc1 produced identical:

- concordant, normally discordant and reverse-discordant counts;
- optimal ages and their stored statistics;
- 200-node goodness-of-fit curves.

The ensemble catalogue changed, as expected, because v2.1.0-rc1 contains the
new broad-crest qualification and support accounting. The unchanged optimal
results were:

| Sample | Concordant | Discordant | Reverse discordant | Optimal age (Ma) |
| --- | ---: | ---: | ---: | ---: |
| 2000082044 | 20 | 20 | 2 | 181.8141 |
| 78206019 | 9 | 19 | 1 | 282.2663 |
| 92969082 | 4 | 31 | 0 | 493.2161 |

The executable benchmark and the two JSON result records are stored in this
directory.

## Automated tests and builds

The local suite passed: 43 tests in 188.10 seconds.

The release workflow now runs its Linux test job and Windows, Intel-macOS and
Apple-silicon-macOS test/build jobs on `release/**` branches. Public release
assets are still created only from version tags.

