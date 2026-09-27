"""CDC configuration constants.

This module exposes both the fixed ensemble-tuning parameters and the optional
diagnostic-output routing used by the GUI and replay tools.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _env_bool(name: str, default: str = "0") -> bool:
    try:
        return bool(int(os.environ.get(name, default)))
    except Exception:
        return bool(int(default))


# ====================== ENSEMBLE PARAMETERS  ======================
# Fixed parameters for the CDC ensemble pipeline.
# Not user-configurable. Nine of these were varied in the threshold
# robustness analysis reported in the paper and are marked "tested".
# The remainder control smoothing, voting geometry, or noise floors
# and do not independently decide whether a peak enters the catalogue.

# --------------- Ensemble curve smoothing ---------------
SMOOTH_FRAC: float = 0.01    # Smoothing kernel as fraction of grid length. Not swept (affects curve shape only).

# --------------- Nearby-peak merging ---------------
MERGE_NEARBY_PEAKS: bool = False              # Not swept.
PLATEAU_DEDUPE: bool = True                   # Not swept.
PLATEAU_DEDUPE_RADIUS_STEPS: float = 2.0      # Not swept.
PLATEAU_DEDUPE_MIN_OVERLAP_FRAC: float = 0.60 # Not swept.

# --------------- Per-run peak detection ---------------
PER_RUN_PROM_FRAC: float = 0.06  # Tested.
PER_RUN_MIN_DIST: int = 3        # Tested.
PER_RUN_MIN_WIDTH: int = 3       # Tested.

# --------------- Ensemble peak acceptance ---------------
FD_DIST_FRAC: float = 0.10       # Tested.
FP_PROM_FRAC: float = 0.10       # Tested.
FW_WIN_FRAC: float = 0.10        # Not swept.
FR_RUN_REL: float = 0.25         # Not swept.
FS_SUPPORT: float = 0.10         # Tested.
RMIN_RUNS: int = 5               # Tested.
FV_VALLEY_FRAC: float = 0.50     # Tested.
ENS_DELTA_MIN: float = 0.05      # Tested.
MONO_DY_EPS_FRAC: float = 0.03   # Not swept.
MONO_MAX_TURNS: int = 0          # Not swept.
COARSE_SIGMA_GRID_FRAC: float = 0.03   # Coarse ensemble smoothing used for major-mode grouping.
DEGENERATE_CI_GRID_FRAC: float = 0.75  # Treat sub-grid intervals narrower than this as degenerate.

# --------------- Reverse-discordance detection ---------------
REV_TOL_Y: float = 1e-5   # 207Pb/206Pb noise tolerance. Not swept.
REV_TOL_X: float = 1e-6   # 238U/206Pb noise tolerance. Not swept.

# --------------- Display ---------------
CATALOGUE_SURFACE: str = "PEN"  # Default display surface ("PEN" or "RAW"). Default to PEN.


# ====================== OUTPUT / DIAGNOSTICS ======================

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
    "per_run_median_s", "per_run_p95_s", "rss_peak_mb", "python", "numpy",
]
