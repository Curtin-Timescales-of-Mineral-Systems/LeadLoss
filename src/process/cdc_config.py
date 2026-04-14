from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


# ======================  ALGORITHM PARAMETERS  ======================
# Fixed parameters for the CDC ensemble peak-picking pipeline.
# These values were calibrated against synthetic and natural zircon
# datasets and are documented in the accompanying publication.
# They are not user-configurable.

# --------------- Ensemble curve smoothing ---------------
# The per-run KS goodness surfaces are aggregated into a median ensemble
# curve. A light Gaussian smooth suppresses single-node noise without
# broadening real peaks. SMOOTH_FRAC sets the kernel width as a fraction
# of the age grid length (e.g. 0.01 = 1% of grid nodes).
# SMOOTH_MA is an alternative fixed-width mode (Ma); 0 disables it.
SMOOTH_MA: float = 0.0
SMOOTH_FRAC: float = 0.01

# --------------- Nearby-peak merging ---------------
# When False, nearby peaks on the ensemble curve are preserved as
# separate catalogue entries rather than merged. Plateau deduplication
# (below) still collapses truly redundant picks on the same flat crest.
MERGE_NEARBY_PEAKS: bool = False

# Peaks whose CIs overlap by >= 60% within a 2-grid-step radius are
# treated as duplicates of the same geological event.
PLATEAU_DEDUPE: bool = True
PLATEAU_DEDUPE_RADIUS_STEPS: float = 2.0
PLATEAU_DEDUPE_MIN_OVERLAP_FRAC: float = 0.60

# --------------- Per-run peak detection ---------------
# Each MC run's goodness curve is independently peak-picked to build
# per-run "votes" for ensemble peaks. These gates control which local
# maxima qualify as per-run peaks.
#   PROM_FRAC : minimum prominence as a fraction of the run's dynamic range
#   MIN_DIST  : minimum separation between peaks (grid nodes)
#   MIN_WIDTH : minimum peak width at half-prominence (grid nodes)
PER_RUN_PROM_FRAC: float = 0.06
PER_RUN_MIN_DIST: int = 3
PER_RUN_MIN_WIDTH: int = 3

# --------------- Ensemble peak acceptance ---------------
# Candidate peaks on the median ensemble curve must pass these gates
# to enter the final catalogue.
#
#   FH_HEIGHT_FRAC : minimum height as fraction of tallest peak (0 = disabled)
#   FD_DIST_FRAC   : minimum separation between peaks (fraction of grid length)
#   FP_PROM_FRAC   : minimum prominence as fraction of ensemble dynamic range
#   FW_WIN_FRAC    : half-width of the per-run vote window (fraction of grid)
#   FR_RUN_REL     : per-run relative height gate — a run only "votes" for a
#                    peak if its local maximum exceeds this fraction of the
#                    run's own dynamic range
#   FS_SUPPORT     : minimum fraction of MC runs that must vote for a peak
#   RMIN_RUNS      : absolute minimum number of supporting runs
#   FV_VALLEY_FRAC : two adjacent peaks are merged if the valley between them
#                    is shallower than this fraction of the smaller prominence
#   ENS_DELTA_MIN  : minimum dynamic range (max - min) of the ensemble curve
#                    to attempt peak picking at all; below this the surface
#                    is effectively flat and no peaks are reported
#   MONO_DY_EPS_FRAC : tolerance for classifying a surface as monotonic
#   MONO_MAX_TURNS   : max direction changes allowed before a surface is
#                      considered non-monotonic (0 = strict)
FH_HEIGHT_FRAC: float = 0.00
FD_DIST_FRAC: float = 0.10
FP_PROM_FRAC: float = 0.10
FW_WIN_FRAC: float = 0.10
FR_RUN_REL: float = 0.25
FS_SUPPORT: float = 0.10
RMIN_RUNS: int = 5
FV_VALLEY_FRAC: float = 0.50
ENS_DELTA_MIN: float = 0.05
MONO_DY_EPS_FRAC: float = 0.03
MONO_MAX_TURNS: int = 0

# --------------- CI span guard ---------------
# Reject catalogue peaks whose vote-percentile interval is too wide to
# represent a well-resolved age. A peak is rejected when the CI span
# exceeds max(CI_SPAN_ABS_CAP_MA, CI_SPAN_FRAC_CAP * peak_age). This
# catches flat-basin pathologies (e.g. sparse concordant set producing
# noisy per-run curves, or a near-flat ensemble surface outside a single
# young-age spike) where per-run argmins scatter across a wide basin and
# the percentile interval balloons far beyond any defensible run-to-run
# reproducibility claim.
CI_SPAN_ABS_CAP_MA: float = 200.0
CI_SPAN_FRAC_CAP: float = 1.0

# --------------- Reverse-discordance detection ---------------
# Spots plotting above and to the left of concordia in Tera-Wasserburg
# space are flagged as reverse-discordant and excluded from the MC loop.
# These tolerances (in ratio units) prevent floating-point noise on
# near-concordant points from triggering false reverse flags.
REV_TOL_Y: float = 1e-5   # 207Pb/206Pb tolerance
REV_TOL_X: float = 1e-6   # 238U/206Pb tolerance

# --------------- Display ---------------
# Which goodness surface to show in the GUI by default.
# "PEN" = penalised (accounts for invalid ages), "RAW" = unpenalised KS D.
CATALOGUE_SURFACE: str = "PEN"


# ======================  OUTPUT / EXPORT CONFIG  ======================
# These remain configurable via environment variables for batch workflows.

def _env_bool(name: str, default: str = "0") -> bool:
    try:
        return bool(int(os.environ.get(name, default)))
    except Exception:
        return bool(int(default))

_ks_root = os.environ.get("CDC_KS_EXPORT_DIR", "").strip()
KS_EXPORT_ROOT: Optional[Path] = Path(_ks_root).expanduser() if _ks_root else None

CDC_WRITE_OUTPUTS: bool = _env_bool("CDC_WRITE_OUTPUTS", "0")
CDC_ENABLE_RUNLOG: bool = _env_bool("CDC_ENABLE_RUNLOG", "0")

if "CDC_TIMING_MODE" in os.environ:
    TIMING_MODE: bool = _env_bool("CDC_TIMING_MODE", "0")
else:
    TIMING_MODE = not CDC_WRITE_OUTPUTS

EXP_TAG: str = os.environ.get("CDC_EXP_TAG", "").strip()
OUT_ROOT: str = os.environ.get("CDC_OUT_DIR", "").strip()

_DEFAULT_ROOT = Path(os.environ.get("CDC_DEFAULT_OUT_DIR", str(Path.home() / "LeadLossOutputs"))).expanduser()

_root = Path(OUT_ROOT).expanduser() if OUT_ROOT else _DEFAULT_ROOT
_stem = "ensemble_catalogue"

BASE_CATALOGUE: Path = _root / (f"{_stem}_{EXP_TAG}" if EXP_TAG else _stem)
CATALOGUE_CSV_PEN: Path = BASE_CATALOGUE.with_suffix(".csv")
CATALOGUE_CSV_RAW: Path = BASE_CATALOGUE.with_name(BASE_CATALOGUE.name + "_np").with_suffix(".csv")

RUNLOG: Path = _root / (f"runtime_log_{EXP_TAG}.csv" if EXP_TAG else "runtime_log.csv")
DIAG_DIR: Path = _root / (f"diag_ks_{EXP_TAG}" if EXP_TAG else "diag_ks")

RUN_FIELDS = [
    "method", "phase", "sample", "tier", "R", "n_grid", "elapsed_s",
    "per_run_median_s", "per_run_p95_s", "rss_peak_mb", "python", "numpy"
]
