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

# --------------- Reverse-discordance detection ---------------
REV_TOL_Y: float = 1e-5   # 207Pb/206Pb noise tolerance. Not swept.
REV_TOL_X: float = 1e-6   # 238U/206Pb noise tolerance. Not swept.

# --------------- Display ---------------
CATALOGUE_SURFACE: str = "PEN"  # Default display surface ("PEN" or "RAW"). Default to PEN.