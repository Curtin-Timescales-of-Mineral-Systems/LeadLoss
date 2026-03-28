from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _env_float(name: str, default: str) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _env_int(name: str, default: str) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return int(default)


def _env_bool(name: str, default: str = "0") -> bool:
    """Environment bools are read as 0/1 for robustness."""
    try:
        return bool(int(os.environ.get(name, default)))
    except Exception:
        return bool(int(default))


def _env_choice(name: str, default: str, choices: tuple[str, ...]) -> str:
    """Return env value normalised to upper-case, constrained to choices."""
    raw = str(os.environ.get(name, default)).strip().upper()
    allowed = tuple(str(c).strip().upper() for c in choices)
    if raw in allowed:
        return raw
    return str(default).strip().upper()


PROFILE: str = _env_choice("CDC_PROFILE", "CUSTOM", ("CUSTOM", "PAPER", "EXPLORATORY"))


def _profile_default(name: str, custom: str, paper: Optional[str] = None, exploratory: Optional[str] = None) -> str:
    """
    Return a string default for env readers based on CDC_PROFILE.
    Explicit env vars for `name` always win.
    """
    if name in os.environ:
        return str(os.environ[name])
    if PROFILE == "PAPER" and paper is not None:
        return str(paper)
    if PROFILE == "EXPLORATORY" and exploratory is not None:
        return str(exploratory)
    return str(custom)


# ======================  EXPOSED KNOBS  ======================

# Smoothing: either a fixed Gaussian width in Ma (SMOOTH_MA > 0),
# or a fraction of grid nodes (SMOOTH_FRAC) if SMOOTH_MA <= 0.
SMOOTH_MA: float = _env_float("CDC_SMOOTH_MA", "0.0")
SMOOTH_FRAC: float = _env_float("CDC_SMOOTH_FRAC", "0.01")

# If 0, disable the conservative peak-merge stages so nearby peaks are preserved.
MERGE_NEARBY_PEAKS: bool = _env_bool(
    "CDC_MERGE_NEARBY_PEAKS",
    _profile_default("CDC_MERGE_NEARBY_PEAKS", "0", paper="0", exploratory="0"),
)

# Clustered per-sample catalogue branch. This is only used on clustering-enabled
# branches; the baseline no-clustering branch simply never references it.
USE_CLUSTER_CATALOGUE: bool = _env_bool("CDC_USE_CLUSTER_CATALOGUE", "1")

# In no-merge mode, still collapse near-identical picks on the same flat crest.
PLATEAU_DEDUPE: bool = _env_bool("CDC_PLATEAU_DEDUPE", "1")
PLATEAU_DEDUPE_RADIUS_STEPS: float = _env_float("CDC_PLATEAU_DEDUPE_RADIUS_STEPS", "2.0")
PLATEAU_DEDUPE_MIN_OVERLAP_FRAC: float = _env_float("CDC_PLATEAU_DEDUPE_MIN_OVERLAP_FRAC", "0.60")

# Per-run peak detector gates (used inside build_ensemble_catalogue for per-run peaks)
PER_RUN_PROM_FRAC: float = _env_float("CDC_PER_RUN_PROM_FRAC", "0.06")
PER_RUN_MIN_DIST: int = _env_int("CDC_PER_RUN_MIN_DIST", "3")
PER_RUN_MIN_WIDTH: int = _env_int("CDC_PER_RUN_MIN_WIDTH", "3")

# Ensemble peak gates – “conservative” thresholds used in the paper
FH_HEIGHT_FRAC: float = _env_float("CDC_FH_HEIGHT_FRAC", "0.00")   # disabled by default
FD_DIST_FRAC: float = _env_float("CDC_FD_DIST_FRAC", "0.10")       # min peak separation in nodes (fraction of grid)
FP_PROM_FRAC: float = _env_float("CDC_FP_PROM_FRAC", "0.10")       # min ensemble prominence as a fraction of Δ
FW_WIN_FRAC: float = _env_float("CDC_FW_WIN_FRAC", "0.10")         # half-width of vote window as fraction of grid
FR_RUN_REL: float = _env_float("CDC_FR_RUN_REL", "0.25")           # run-level relative height gate (fraction of run’s dynamic range)
FS_SUPPORT: float = _env_float(
    "CDC_FS_SUPPORT",
    _profile_default("CDC_FS_SUPPORT", "0.10", paper="0.10", exploratory="0.08"),
)           # minimum support fraction across runs
RMIN_RUNS: int = _env_int("CDC_RMIN_RUNS", "5")                    # minimum number of runs contributing
FV_VALLEY_FRAC: float = _env_float("CDC_FV_VALLEY_FRAC", "0.50")   # shallow-valley merge threshold
ENS_DELTA_MIN: float = _env_float("CDC_ENS_DELTA_MIN", "0.05")     # min ensemble dynamic range to attempt peak picking
MONO_DY_EPS_FRAC: float = _env_float("CDC_MONO_DY_EPS_FRAC", "0.03")
MONO_MAX_TURNS: int = _env_int("CDC_MONO_MAX_TURNS", "0")

# Reverse-discordance geometry tolerances (TW space)
REV_TOL_Y: float = _env_float("CDC_REV_TOL_Y", "1e-5")   # vertical tolerance in 207Pb/206Pb
REV_TOL_X: float = _env_float("CDC_REV_TOL_X", "1e-6")   # horizontal tolerance in 238U/206Pb

# ======================  OUTPUT / UI DEFAULTS  ======================

# NEW: optional legacy KS export directory (for the paper figure)
_ks_root = os.environ.get("CDC_KS_EXPORT_DIR", "").strip()
KS_EXPORT_ROOT: Optional[Path] = Path(_ks_root).expanduser() if _ks_root else None

# Which catalogue surface to show in UI by default (no effect on detection)
UI_SURFACE: str = os.environ.get("CDC_UI_SURFACE", "PEN").strip().upper()
CATALOGUE_SURFACE: str = os.environ.get("CDC_CATALOGUE_SURFACE", "PEN").strip().upper()

if UI_SURFACE not in {"RAW", "PEN"}:
    UI_SURFACE = "PEN"
if CATALOGUE_SURFACE not in {"RAW", "PEN"}:
    CATALOGUE_SURFACE = UI_SURFACE

# Whether to write CSV/NPZ diagnostics and run-log
CDC_WRITE_OUTPUTS: bool = _env_bool("CDC_WRITE_OUTPUTS", "0")
CDC_ENABLE_RUNLOG: bool = _env_bool("CDC_ENABLE_RUNLOG", "0")

# Timing mode: when writing outputs, disable timing-only mode
if "CDC_TIMING_MODE" in os.environ:
    TIMING_MODE: bool = _env_bool("CDC_TIMING_MODE", "0")
else:
    TIMING_MODE = not CDC_WRITE_OUTPUTS

# Optional routing to avoid overwriting previous runs
EXP_TAG: str = os.environ.get("CDC_EXP_TAG", "").strip()
OUT_ROOT: str = os.environ.get("CDC_OUT_DIR", "").strip()

# IMPORTANT:
# - Hard-coded absolute paths (e.g., /Users/...) make the repo non-portable.
# - We therefore default outputs to a folder under the user's home directory,
#   *but only* if outputs are enabled.
_DEFAULT_ROOT = Path(os.environ.get("CDC_DEFAULT_OUT_DIR", str(Path.home() / "LeadLossOutputs"))).expanduser()

_root = Path(OUT_ROOT).expanduser() if OUT_ROOT else _DEFAULT_ROOT
_stem = "ensemble_catalogue"

BASE_CATALOGUE: Path = _root / (f"{_stem}_{EXP_TAG}" if EXP_TAG else _stem)
CATALOGUE_CSV_PEN: Path = BASE_CATALOGUE.with_suffix(".csv")  # penalised catalogue
CATALOGUE_CSV_RAW: Path = BASE_CATALOGUE.with_name(BASE_CATALOGUE.name + "_np").with_suffix(".csv")  # raw / non-penalised

RUNLOG: Path = _root / (f"runtime_log_{EXP_TAG}.csv" if EXP_TAG else "runtime_log.csv")
DIAG_DIR: Path = _root / (f"diag_ks_{EXP_TAG}" if EXP_TAG else "diag_ks")

RUN_FIELDS = [
    "method", "phase", "sample", "tier", "R", "n_grid", "elapsed_s",
    "per_run_median_s", "per_run_p95_s", "rss_peak_mb", "python", "numpy"
]
