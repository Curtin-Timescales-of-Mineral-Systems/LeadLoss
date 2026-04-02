from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


# ======================  ALGORITHM PARAMETERS  ======================
# These are the published defaults. They are not user-configurable.

# Smoothing: fraction of grid nodes for Gaussian smoothing kernel
SMOOTH_MA: float = 0.0
SMOOTH_FRAC: float = 0.01

# Peak merging
MERGE_NEARBY_PEAKS: bool = False

# Plateau deduplication (collapse near-identical picks on the same flat crest)
PLATEAU_DEDUPE: bool = True
PLATEAU_DEDUPE_RADIUS_STEPS: float = 2.0
PLATEAU_DEDUPE_MIN_OVERLAP_FRAC: float = 0.60

# Per-run peak detector gates
PER_RUN_PROM_FRAC: float = 0.06
PER_RUN_MIN_DIST: int = 3
PER_RUN_MIN_WIDTH: int = 3

# Ensemble peak gates
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

# Reverse-discordance geometry tolerances (TW space)
REV_TOL_Y: float = 1e-5
REV_TOL_X: float = 1e-6

# Display surface
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
