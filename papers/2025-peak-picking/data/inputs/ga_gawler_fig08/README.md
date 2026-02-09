# Fig. 08 (Gawler Craton) input data

This folder contains CSV extracts used to generate the illustrative natural-data example (Fig. 08).

## Files (inputs in this folder)

- `ga_shrimp_analyses_gawler_gneiss_sample_2008371041_accessed_2026-02-06.csv`  
  Spot-level SHRIMP analytical ratios and uncertainties (used for the Wetherill concordia inset).

- `ga_shrimp_sample_group_ages_gawler_gneiss_sample_2008371041_accessed_2026-02-06.csv`  
  Interpreted sample “group ages” / interpreted results (used for the interpreted ages reported in the text).

## CDC outputs used for Fig. 08

### A) Peak ages and 95% confidence intervals reported in the text

The Pb-loss peak ages and 95% confidence intervals reported for Fig. 08 (e.g., ~815 Ma and ~269 Ma) are taken from the manuscript bundle’s CDC ensemble catalogue:

- `papers/2025-peak-picking/data/derived/ensemble_catalogue.csv`

Relevant columns in `ensemble_catalogue.csv`:

- `sample` — identifier used by the CDC catalogue for the dataset (string).
- `peak_no` — peak index within that sample (integer; 1, 2, ...).
- `age_ma` — peak age estimate (Ma).
- `ci_low` — lower bound of the 95% confidence interval (Ma).
- `ci_high` — upper bound of the 95% confidence interval (Ma).
- `support` — peak support metric as exported by the CDC workflow.

### B) Curves/bands used to plot the Fig. 08 main panel

The Monte Carlo goodness curves and peak confidence-interval bands plotted in Fig. 08 are based on NPZ diagnostics outputs. To keep the Git repository lightweight, these NPZ files are distributed as a tarball:

- `papers/2025-peak-picking/data/derived/ks_diagnostics_gawler_npz.tar.gz`

Extracting that tarball creates:

- `papers/2025-peak-picking/data/derived/ks_diagnostics_gawler/`

which contains (at minimum):

- `sample_runs_S.npz` — Monte Carlo run-level goodness curves used for the grey curves and interquartile envelope.
- `sample_ensemble_surfaces.npz` — ensemble curve and peak summary arrays used for CI bands / peak markers.

If the extracted folder is missing, you can extract it manually (from the repo root):

```bash
tar -xzf papers/2025-peak-picking/data/derived/ks_diagnostics_gawler_npz.tar.gz \
  -C papers/2025-peak-picking/data/derived
```

Note:
- The extracted folder `ks_diagnostics_gawler/` is gitignored; the tarball is version-controlled.

## Source datasets (Geoscience Australia)

These CSV extracts were downloaded from Geoscience Australia’s Geochronology and Isotopes Web Feature Service (WFS) layers.

SHRIMP analyses layer (WFS CSV):  
https://services.ga.gov.au/gis/geochronology-isotopes/wfs?request=GetFeature&service=WFS&version=1.1.0&typeName=isotope:shrimp_analyses&outputFormat=csv

Metadata record (landing page):  
https://portal.ga.gov.au/metadata/33d9d8e7-f310-43d5-b50a-7fbdd03cab43

SHRIMP interpreted results (sample group ages; WFS CSV):  
https://services.ga.gov.au/gis/geochronology-isotopes/wfs?request=GetFeature&service=WFS&version=1.1.0&typeName=isotope:shrimp_sample_group_ages&outputFormat=csv

Metadata record (landing page):  
https://portal.ga.gov.au/metadata/ccbaa8eb-60b0-432d-b2c5-b13ce5eda30c

Accessed: 2026-02-06

## Subsetting and pre-filters (performed prior to export)

These files contain only the records required for Fig. 08:

- Sample identifier: `GASampleID = 2008371041`
- Rock type: gneiss (from the interpreted-results layer)
- Prior to export we removed analyses with **1σ relative uncertainty ≥ 10%** in either:
  - `c4 238U/206Pb 1sigma (%)`, or
  - `c4 207Pb/206Pb 1sigma (%)`.

Concordance/discordance filtering (10% discordance threshold) is applied by the manuscript scripts, as described in the paper.

## Column definitions (as used in this manuscript)

The tables below explain the columns used by the Fig. 08 scripts and/or referenced in the manuscript text.

### A) Spot-level analytical ratios: `ga_shrimp_analyses_gawler_gneiss_sample_2008371041_accessed_2026-02-06.csv`

| Column | Meaning | Units / notes |
|---|---|---|
| `SampleID` | Geoscience Australia sample identifier for the analysis | integer (here, 2008371041) |
| `c4 238U/206Pb` | Radiogenic (204Pb-corrected) 238U/206Pb | ratio |
| `c4 238U/206Pb 1sigma (%)` | 1σ relative uncertainty on radiogenic (204Pb-corrected) 238U/206Pb | percent |
| `c4 207Pb/206Pb` | Radiogenic (204Pb-corrected) 207Pb/206Pb | ratio |
| `c4 207Pb/206Pb 1sigma (%)` | 1σ relative uncertainty on radiogenic (204Pb-corrected) 207Pb/206Pb | percent |
| `c4 207Pb/235U` | Radiogenic (204Pb-corrected) 207Pb/235U | ratio |
| `c4 207Pb/235U 1sigma (%)` | 1σ relative uncertainty on radiogenic (204Pb-corrected) 207Pb/235U | percent |
| `c4 206Pb/238U` | Radiogenic (204Pb-corrected) 206Pb/238U | ratio |
| `c4 206Pb/238U 1sigma (%)` | 1σ relative uncertainty on radiogenic (204Pb-corrected) 206Pb/238U | percent |
| `rho` | Correlation coefficient (ρ) between the uncertainties of `c4 207Pb/235U` and `c4 206Pb/238U` | used for error ellipses on Wetherill concordia |

Notes:
- The inset concordia plot uses the Wetherill ratios (`c4 207Pb/235U`, `c4 206Pb/238U`) plus their 1σ uncertainties and ρ to draw 95% error ellipses.

### B) Interpreted results (sample group ages): `ga_shrimp_sample_group_ages_gawler_gneiss_sample_2008371041_accessed_2026-02-06.csv`

| Column | Meaning | Units / notes |
|---|---|---|
| `SAMPLE_ID` | Geoscience Australia sample identifier | integer |
| `IGSN` | International Geo Sample Number | string |
| `SAMPLE_PID` | Persistent identifier / URI for the sample record | URL |
| `STRAT_UNIT_NAME` | Stratigraphic unit name | string |
| `STRAT_UNIT_PID` | Persistent identifier / URI for stratigraphic unit | URL |
| `INFORMAL_STRAT_UNIT_NAME` | Informal stratigraphic unit name | string |
| `QUALIFIED_LITHOLOGY` | Lithology description | string |
| `LITHOLOGY` | Lithology (general) | string |
| `SAMPLE_TYPE` | Sample type | string |
| `ENO` | GA internal field (as provided in the WFS export) | numeric |
| `SAMPLING_FEATURE_NAME` | Name/identifier of the sampling feature | string |
| `SAMPLING_FEATURE_TYPE` | Sampling feature type | string |
| `SAMPLING_FEATURE_PID` | Persistent identifier / URI for the sampling feature | URL |
| `GEOLOGICAL_PROVINCE` | Geological province | string |
| `GEOREGION` | Geographic/geologic region | string |
| `STATE` | Australian state | string |
| `COUNTRY` | Country code | string |
| `SAMPLE_LAT_GDA94` | Latitude (GDA94 datum) | degrees |
| `SAMPLE_LONG_GDA94` | Longitude (GDA94 datum) | degrees |
| `SAMPLE_LOCATION_ACCURACY_M` | Location accuracy | metres |
| `SAMPLE_ELEV_AHD_M` | Elevation (AHD) | metres |
| `LOCATION_METHOD` | Location method | string |
| `SAMPLING_DATE` | Sampling date | string/date |
| `SAMPLE_ORIGINATOR` | Sample originator | string |
| `MINERAL` | Mineral analysed | string |
| `INTERPRETATION_SET_NO` | Interpretation set identifier | integer |
| `GROUP_RANK` | Group rank within the interpretation set | string (e.g., A, B, C…) |
| `GROUP_LABEL` | Group label | string |
| `GEOLOGICAL_ATTRIBUTION` | Geological attribution assigned to group | string |
| `SPOTS_IN_GROUP` | Number of spots/analyses in group | integer |
| `ROCK_EVENT_1_AGE_MA` | Interpreted event age | Ma |
| `ROCK_EVENT_1_ERROR_MA` | Uncertainty on interpreted event age | Ma |
| `ROCK_EVENT_1_UNCERTAINTY` | Uncertainty type (as given by GA) | string (e.g., “95% confidence”) |
| `GROUP_DESCRIPTION` | Narrative description of the group | string |
| `RESULT_DETAILS` | Narrative details of the reported result | string |
| `PUBLICATION` | Publication/source attribution | string |
| `PUBLICATION_LINK` | Link to publication/source | URL |
| `SAMPLING_FEATURE_LOCATION` | Service-export field | not used in this manuscript |
