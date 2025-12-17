import time
from enum import Enum
from pathlib import Path
import csv, sys, platform, resource, os, re, zlib
from typing import List, Dict

import numpy as np
import scipy as sp

from model.monteCarloRun import MonteCarloRun
from process import calculations
from model.settings.calculation import DiscordanceClassificationMethod
from utils import config
from process.discordantClustering import (
    find_discordant_clusters,
    _labels_from_this_run,
    _stack_min_across_clusters,
    lower_intercept_proxy,
    _soft_accept_labels,
    _adaptive_gates,
    stack_goodness_by_cluster,
)
from utils.peakHelpers import fmt_peak_stats
from process.ensemble import robust_ensemble_curve, build_ensemble_catalogue

# ======================  EXPOSED KNOBS  ======================

# Smoothing: either a fixed Gaussian width in Ma (SMOOTH_MA > 0),
# or a fraction of grid nodes (SMOOTH_FRAC) if SMOOTH_MA <= 0.
SMOOTH_MA   = float(os.environ.get("CDC_SMOOTH_MA", "0.0"))
SMOOTH_FRAC = float(os.environ.get("CDC_SMOOTH_FRAC", "0.01"))

# Whether to build separate ensemble peaks per discordant cluster.
# For most real datasets you probably want False so the catalogue
# matches the plotted global ensemble curve.
USE_CLUSTER_CATALOGUE = False

# Per-run peak detector gates (used inside build_ensemble_catalogue for per-run peaks)
PER_RUN_PROM_FRAC = 0.10
PER_RUN_MIN_DIST  = 5
PER_RUN_MIN_WIDTH = 5

# Ensemble peak gates – “conservative” thresholds used in the paper
FH_HEIGHT_FRAC = 0.50   # keep peaks whose crest is ≥50% of the tallest one
FD_DIST_FRAC   = 0.10   # min peak separation in nodes (fraction of grid)
FP_PROM_FRAC   = 0.10   # min ensemble prominence as a fraction of Δ 0.05
FW_WIN_FRAC    = 0.10   # half-width of vote window as fraction of grid
FR_RUN_REL     = 0.25   # run-level relative height gate (fraction of run’s dynamic range)
FS_SUPPORT     = 0.10   # minimum support fraction across runs
RMIN_RUNS      = 5      # minimum number of runs contributing 
FV_VALLEY_FRAC = 0.50   # shallow-valley merge threshold
ENS_DELTA_MIN  = 0.05   # min ensemble dynamic range to attempt peak picking

# Reverse-discordance geometry tolerances (TW space)
REV_TOL_Y = 1e-5   # vertical tolerance in 207Pb/206Pb
REV_TOL_X = 1e-6   # horizontal tolerance in 238U/206Pb

# ======================  Output / diagnostics paths  ======================

DIAG_DIR = Path("/Users/lucymathieson/Desktop/Desktop - Lucy’s MacBook Pro - 1/LeadLoss-2/diag_ks")
RUNLOG   = Path("/Users/lucymathieson/Desktop/Peak-Picking-Manuscript-Python/runtime_log.csv")

_BASE_CATALOGUE   = Path("/Users/lucymathieson/Desktop/Peak-Picking-Manuscript-Python/ensemble_catalogue")
CATALOGUE_CSV_PEN = _BASE_CATALOGUE.with_suffix(".csv")  # penalised catalogue
CATALOGUE_CSV_RAW = _BASE_CATALOGUE.with_name(_BASE_CATALOGUE.name + "_np").with_suffix(".csv")  # raw / non-penalised

# Which catalogue surface to show in UI by default (no effect on detection)
UI_SURFACE        = os.environ.get("CDC_UI_SURFACE", "PEN").strip().upper()
CATALOGUE_SURFACE = os.environ.get("CDC_CATALOGUE_SURFACE", "PEN").strip().upper()

if UI_SURFACE not in {"RAW", "PEN"}:
    UI_SURFACE = "PEN"
if CATALOGUE_SURFACE not in {"RAW", "PEN"}:
    CATALOGUE_SURFACE = UI_SURFACE

# Whether to write CSV/NPZ diagnostics and run-log
CDC_WRITE_OUTPUTS = bool(int(os.environ.get("CDC_WRITE_OUTPUTS", "0")))
CDC_ENABLE_RUNLOG = bool(int(os.environ.get("CDC_ENABLE_RUNLOG", "0")))

# Timing mode: when writing outputs, disable timing-only mode
if "CDC_TIMING_MODE" in os.environ:
    TIMING_MODE = bool(int(os.environ["CDC_TIMING_MODE"]))
else:
    TIMING_MODE = not CDC_WRITE_OUTPUTS

# Optional routing to avoid overwriting previous runs
EXP_TAG  = os.environ.get("CDC_EXP_TAG", "").strip()
OUT_ROOT = os.environ.get("CDC_OUT_DIR", "").strip()
if EXP_TAG or OUT_ROOT:
    root = Path(OUT_ROOT).expanduser() if OUT_ROOT else _BASE_CATALOGUE.parent
    stem = "ensemble_catalogue"
    _BASE_CATALOGUE   = root / (f"{stem}_{EXP_TAG}" if EXP_TAG else stem)
    CATALOGUE_CSV_PEN = _BASE_CATALOGUE.with_suffix(".csv")
    CATALOGUE_CSV_RAW = _BASE_CATALOGUE.with_name(_BASE_CATALOGUE.name + "_np").with_suffix(".csv")
    RUNLOG            = root / (f"runtime_log_{EXP_TAG}.csv" if EXP_TAG else "runtime_log.csv")
    DIAG_DIR          = root / (f"diag_ks_{EXP_TAG}" if EXP_TAG else "diag_ks")

def _ensure_output_dirs():
    """Create output directories for diagnostics if CDC_WRITE_OUTPUTS is enabled."""
    if not CDC_WRITE_OUTPUTS:
        return
    CATALOGUE_CSV_PEN.parent.mkdir(parents=True, exist_ok=True)
    CATALOGUE_CSV_RAW.parent.mkdir(parents=True, exist_ok=True)
    RUNLOG.parent.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

if CDC_WRITE_OUTPUTS:
    _ensure_output_dirs()

RUN_FIELDS = [
    "method", "phase", "sample", "tier", "R", "n_grid", "elapsed_s",
    "per_run_median_s", "per_run_p95_s", "rss_peak_mb", "python", "numpy"
]

# ======================  UTIILTIES  ======================

def _write_runlog(row: dict):
    if not CDC_ENABLE_RUNLOG:
        return
    RUNLOG.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RUNLOG.exists()
    with RUNLOG.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RUN_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)

def _seed_from_name(name: str, base: int = 42) -> int:
    return (base ^ zlib.adler32((name or "").encode("utf-8"))) & 0xFFFFFFFF

def _reset_csv(path: Path, header: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        fh.write(header + "\n")

def _rss_mb():
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru/(1024*1024.0) if sys.platform == "darwin" else ru/1024.0

def _safe_prefix(name: str) -> str:
    s = str(name or "").strip()
    s = re.sub(r"[^\w\-.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_.")
    return s or "sample"
            
def _infer_tier(sample_name: str) -> str:
    s = (sample_name or "").strip()
    return s[-1].upper() if s and s[-1].isalpha() else ""

def _norm_catalogue(rows):
    out = []
    if isinstance(rows, dict):
        rows = [rows]
    if isinstance(rows, list):
        for i, r in enumerate(rows, 1):
            if isinstance(r, dict):
                out.append(dict(
                    age_ma  = float(r.get("age_ma",  float("nan"))),
                    ci_low  = float(r.get("ci_low",  float("nan"))),
                    ci_high = float(r.get("ci_high", float("nan"))),
                    support = float(r.get("support", float("nan"))),
                    peak_no = int(r.get("peak_no", i)),
                ))
    return out

def _append_catalogue_rows(sample_name, rows, dest_path):
    if not rows:
        return
    dest_path = Path(dest_path)
    write_header = not dest_path.exists()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with dest_path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample","peak_no","age_ma","ci_low","ci_high","support"], extrasaction="ignore")
        if write_header:
            w.writeheader()
        for i, r in enumerate(rows, 1):
            w.writerow(dict(
                sample  = sample_name,
                peak_no = r.get("peak_no", i),
                age_ma  = r["age_ma"],
                ci_low  = r["ci_low"],
                ci_high = r["ci_high"],
                support = r.get("support", float("nan")),
            ))
            
# ======================  DECAY CONSTANTS & TW HELPERS  ======================

LAMBDA_238 = 1.55125e-4
LAMBDA_235 = 9.8485e-4
R_U        = 137.818  # 238U/235U

def _age_ma_from_u238pb206(u):
    try:
        u = float(u)
        if not np.isfinite(u) or u <= 0.0:
            return float("nan")
        return (1.0 / LAMBDA_238) * np.log1p(1.0 / u)
    except Exception:
        return float("nan")

def _age_ma_from_pb207pb206(v):
    try:
        v = float(v)
        if not np.isfinite(v) or v <= 0.0:
            return float("nan")

        lo, hi = 1e-9, 5000.0  # was 0.0; avoid 0/0 at t=0
        def f(t):
            num = np.expm1(LAMBDA_235*t)
            den = np.expm1(LAMBDA_238*t)
            if den == 0.0:
                # limit t→0 of (e^{λ235 t}-1)/(e^{λ238 t}-1) = λ235/λ238
                ratio = LAMBDA_235 / LAMBDA_238
            else:
                ratio = num / den
            return ratio - (v * R_U)

        flo = f(lo)
        for _ in range(60):
            mid = 0.5*(lo+hi)
            fm  = f(mid)
            if np.sign(fm) == np.sign(flo):
                lo, flo = mid, fm
            else:
                hi = mid
        return 0.5*(lo+hi)
    except Exception:
        return float("nan")

def _is_reverse_discordant(u, v, tol_y=REV_TOL_Y, tol_x=REV_TOL_X):
    """
    TW coordinates: x = 238U/206Pb (= u), y = 207Pb/206Pb (= v)

    A discordant point is 'reverse' if EITHER:
      • below the curve at the same x   : v < y_concordia(u) - tol_y
      • left  of the curve at the same y: u < x_concordia(v) - tol_x
    """
    try:
        if not (np.isfinite(u) and np.isfinite(v)) or u <= 0.0 or v <= 0.0:
            return False

        # Below-at-same-x
        v_conc = calculations.pb207pb206_from_u238pb206(float(u))
        if np.isfinite(v_conc) and (float(v) < (v_conc - tol_y)):
            return True

        # Left-of-at-same-y: invert y→t, then t→x on concordia
        t_ma = _age_ma_from_pb207pb206(float(v))
        if np.isfinite(t_ma):
            x_conc = calculations.u238pb206_from_age(float(t_ma) * 1e6)  # expects years
            if np.isfinite(x_conc) and (float(u) < (x_conc - tol_x)):
                return True
    except Exception:
        pass
    return False

# ======================  POPULATION HELPERS ======================

def _concordant_ages_ma(spots):
    """
    Compute approximate ages (Ma) for concordant spots, used for population clustering.

    Tries 207Pb/206Pb age first (robust for old SHRIMP data), falling back to
    238U/206Pb if needed. Returns NaN where neither estimate is finite.
    """
    ages = []
    for s in spots:
        # use 207Pb/206Pb age as "population" age (robust at high t)
        t = _age_ma_from_pb207pb206(s.pbPbValue)
        if np.isfinite(t):
            ages.append(t)
        else:
            # fallback to U-Pb age if needed
            t2 = _age_ma_from_u238pb206(s.uPbValue)
            ages.append(t2 if np.isfinite(t2) else np.nan)
    ages = np.asarray(ages, float)
    return ages

def _cluster_concordant_populations(concordantSpots, max_pops=3):
    """
    Cluster concordant ages into up to max_pops populations using 1-D GMM+BIC.
    Returns labels per concordant spot, number of pops, and pop means (Ma).
    """
    ages = _concordant_ages_ma(concordantSpots)
    mask = np.isfinite(ages)
    ages_f = ages[mask]
    if ages_f.size < 2:
        return np.zeros(len(concordantSpots), int), 1, np.array([np.nan])

    X = ages_f.reshape(-1, 1)
    try:
        from sklearn.mixture import GaussianMixture
        best_gm, best_bic, best_k = None, np.inf, 1
        for k in range(1, min(max_pops, ages_f.size) + 1):
            gm = GaussianMixture(n_components=k, covariance_type="full", random_state=0)
            gm.fit(X)
            bic = gm.bic(X)
            if bic < best_bic:
                best_bic, best_gm, best_k = bic, gm, k
        labels_f = best_gm.predict(X)
        labels = np.zeros(len(concordantSpots), int)
        labels[mask] = labels_f
        means = best_gm.means_.ravel()
        return labels, int(best_k), means
    except Exception:
        # fallback: single population
        return np.zeros(len(concordantSpots), int), 1, np.array([np.nan])

def _assign_discordant_to_populations(discordantSpots, pop_means_ma):
    """
    Assign each discordant spot to the nearest concordant population in age.
    """
    labels = np.zeros(len(discordantSpots), int)
    if pop_means_ma is None or not np.isfinite(pop_means_ma).any():
        return labels
    pops = np.asarray(pop_means_ma, float)
    for i, s in enumerate(discordantSpots):
        t = _age_ma_from_pb207pb206(s.pbPbValue)
        if not np.isfinite(t):
            t = _age_ma_from_u238pb206(s.uPbValue)
        if not np.isfinite(t):
            labels[i] = 0
        else:
            labels[i] = int(np.argmin(np.abs(pops - t)))
    return labels

# ======================  PROGRESS  ======================

TIME_PER_TASK = 0.0

class ProgressType(Enum):
    CONCORDANCE = 0
    SAMPLING    = 1
    OPTIMAL     = 2

def processSamples(signals, samples):
    if CDC_WRITE_OUTPUTS:
        _reset_csv(CATALOGUE_CSV_PEN, "sample,peak_no,age_ma,ci_low,ci_high,support")
        _reset_csv(CATALOGUE_CSV_RAW, "sample,peak_no,age_ma,ci_low,ci_high,support")
        _reset_csv(RUNLOG,            "method,phase,sample,tier,R,n_grid,elapsed_s,per_run_median_s,per_run_p95_s,rss_peak_mb,python,numpy")

    for sample in samples:
        completed, skip_reason = _processSample(signals, sample)
        if not completed and skip_reason:
            signals.skipped(sample.name, skip_reason)

    signals.completed()

def _processSample(signals, sample):
    t0 = time.perf_counter()

    try:
        # 1) Classify concordant/discordant (incl. reverse flags)
        completed, skip_reason = _calculateConcordantAges(signals, sample)
        if not completed:
            return False, skip_reason

        # 2) First pass: MC sampling + ensemble
        completed, skip_reason = _performRimAgeSampling(signals, sample)
        if not completed:
            return False, skip_reason
        
        # >>> NEW: if we used population-split, skip edge-guard entirely <<<
        st = sample.calculationSettings
        if bool(getattr(st, "split_by_concordant_population", False)):
            return True, None

        # 3) Edge-guard: widen once if many per-run optima hug a boundary
        try:
            ages_ma = np.asarray(st.rimAges(), float) / 1e6
            if ages_ma.size >= 2 and sample.monteCarloRuns:
                # robust step estimate
                raw_step = float(np.median(np.diff(ages_ma))) if ages_ma.size >= 2 else 0.0
                if not np.isfinite(raw_step) or raw_step <= 0.0:
                    raw_step = 5.0
                step_ma = raw_step

                opt_ma  = np.array([r.optimal_pb_loss_age for r in sample.monteCarloRuns],
                                   dtype=float) / 1e6
                opt_ma  = opt_ma[np.isfinite(opt_ma)]
                if opt_ma.size:
                    hit_lo    = np.mean(opt_ma <= (ages_ma[0] + step_ma))      # young boundary
                    hit_hi    = np.mean(opt_ma >= (ages_ma[-1] - step_ma))     # old boundary
                    edge_hits = max(hit_lo, hit_hi)

                    if edge_hits > 0.20:
                        span_y        = float(st.maximumRimAge) - float(st.minimumRimAge)
                        expand_y      = 0.20 * span_y
                        widen_younger = (hit_lo >= hit_hi)

                        if widen_younger:
                            new_min = max(0.0, float(st.minimumRimAge) - expand_y)
                            new_max = float(st.maximumRimAge)
                        else:
                            new_min = float(st.minimumRimAge)
                            new_max = float(st.maximumRimAge) + expand_y

                        # keep grid step roughly constant
                        ages_ma_new = np.asarray(st.rimAges(), float) / 1e6
                        raw_step2 = float(np.median(np.diff(ages_ma_new))) if ages_ma_new.size >= 2 else 0.0
                        if not np.isfinite(raw_step2) or raw_step2 <= 0.0:
                            raw_step2 = step_ma or 5.0
                        step_ma2 = raw_step2

                        span_ma_new      = (new_max - new_min) / 1e6
                        safe_step_for_div = step_ma2 if step_ma2 > 1e-9 else max(span_ma_new, 1.0)
                        st.minimumRimAge  = new_min
                        st.maximumRimAge  = new_max
                        st.rimAgesSampled = int(max(3, round(span_ma_new / safe_step_for_div) + 1))

                        # re-run once with widened window
                        sample.monteCarloRuns = []
                        sample.peak_catalogue = []
                        completed, skip_reason = _performRimAgeSampling(signals, sample)
                        if not completed:
                            return False, skip_reason

        except Exception as _edge_guard_err:
            print(f"[CDC] Edge-guard diagnostic failed for {sample.name}: {_edge_guard_err}")


        return True, None

    finally:
        # runtime log (best effort)
        try:
            n_grid = len(sample.calculationSettings.rimAges())
            R_runs = sample.calculationSettings.monteCarloRuns
        except Exception:
            n_grid, R_runs = 0, 0
        _write_runlog(dict(
            method="CDC", phase="e2e_runtime",
            sample=sample.name, tier=_infer_tier(sample.name),
            R=R_runs, n_grid=n_grid,
            elapsed_s=round(time.perf_counter() - t0, 3),
            per_run_median_s="", per_run_p95_s="",
            rss_peak_mb=round(_rss_mb(), 1),
            python=platform.python_version(), numpy=np.__version__,
        ))

def _calculateConcordantAges(signals, sample):
    """
    Classify each valid spot as concordant/discordant and flag reverse discordance.

    - Concordance is determined either by percentage discordance or by error ellipse,
      according to sample.calculationSettings.discordanceClassificationMethod.
    - A spot is marked as reverseDiscordant if it is geometrically reverse in TW space
      and fails the concordance test.

    Emits:
      - ProgressType.CONCORDANCE updates for UI.
      - sample.updateConcordance(concordancy, discordances, reverse_flags).
    """

    sampleNameText = f" for '{sample.name}'" if sample.name else ""
    signals.newTask("Classifying points" + sampleNameText + "...")

    settings   = sample.calculationSettings
    n_spots    = max(1, len(sample.validSpots))
    timePerRow = TIME_PER_TASK / n_spots

    concordancy   = []
    discordances  = []
    reverse_flags = []   # << NEW

    for i, spot in enumerate(sample.validSpots):
        signals.progress(ProgressType.CONCORDANCE, i / n_spots)
        time.sleep(timePerRow)
        if signals.halt():
            signals.cancelled()
            return False, "processing halted by user"

        if settings.discordanceClassificationMethod == DiscordanceClassificationMethod.PERCENTAGE:
            discordance = calculations.discordance(spot.uPbValue, spot.pbPbValue)
            # Concordant if the *magnitude* is under the threshold
            concordant  = abs(discordance) < settings.discordancePercentageCutoff
        else:
            discordance = None
            concordant = calculations.isConcordantErrorEllipse(
                spot.uPbValue,  spot.uPbStDev,
                spot.pbPbValue, spot.pbPbStDev,
                settings.discordanceEllipseSigmas
            )

        is_rev_geom = _is_reverse_discordant(spot.uPbValue, spot.pbPbValue)

        # Only mark reverse if it’s *discordant* and geometrically reverse
        spot.reverseDiscordant = bool(is_rev_geom and not concordant)

        discordances.append(discordance)
        concordancy.append(concordant)

    reverse_flags = [bool(s.reverseDiscordant) for s in sample.validSpots]
    sample.updateConcordance(concordancy, discordances, reverse_flags)

    n_rev = sum(reverse_flags)
    n_fwd = sum(1 for c, r in zip(concordancy, reverse_flags) if (not c) and (not r))
    n_con = sum(1 for c in concordancy if c)

    signals.progress(ProgressType.CONCORDANCE, 1.0, sample.name, concordancy, discordances, reverse_flags)
    return True, None

# ======================  MC Sampling  ======================

def _performSingleRun(settings, run):
    for age in settings.rimAges():
        run.samplePbLossAge(age, settings.dissimilarityTest, settings.penaliseInvalidAges)
    run.calculateOptimalAge()
    run.createHeatmapData(settings.minimumRimAge, settings.maximumRimAge, config.HEATMAP_RESOLUTION)

def _performRimAgeSampling(signals, sample):
    """
    Run Monte Carlo sampling of Pb-loss ages for a single sample.

    - Filters out reverse-discordant spots.
    - Requires ≥1 concordant and ≥3 forward-discordant spots.
    - Draws MC replicates for U/Pb, Pb/Pb for each spot.
    - Optionally clusters discordant grains using lower-intercept proxies.
    - For each MC run, constructs a MonteCarloRun over the Pb-loss grid,
      finds the optimal Pb-loss age, and emits ProgressType.SAMPLING.

    If settings.split_by_concordant_population is True, delegates to
    _performRimAgeSampling_split_by_population instead.
    """
    sample.monteCarloRuns = []
    sample.peak_catalogue = []
    sampleNameText = f" for '{sample.name}'" if sample.name else ""
    signals.newTask("Sampling Pb-loss age distributions" + sampleNameText + "...")

    settings = sample.calculationSettings
    setattr(settings, "timing_mode", TIMING_MODE)
    setattr(settings, "write_outputs", CDC_WRITE_OUTPUTS)

    # NEW: population-aware mode
    if bool(getattr(settings, "split_by_concordant_population", False)):
        return _performRimAgeSampling_split_by_population(signals, sample)

    eligibleSpots   = [s for s in sample.validSpots if not getattr(s, "reverseDiscordant", False)]
    concordantSpots = [s for s in eligibleSpots if s.concordant]
    discordantSpots = [s for s in eligibleSpots if not s.concordant]

    if not concordantSpots:
        return False, "no concordant spots"
    if not discordantSpots:
        return False, "no discordant spots"
    if len(discordantSpots) <= 2:
        return False, "fewer than 3 discordant spots"

    stabilitySamples = int(settings.monteCarloRuns)
    rng = np.random.default_rng(_seed_from_name(sample.name))

    concordantUPbValues  = np.stack([rng.normal(s.uPbValue,  s.uPbStDev,  stabilitySamples) for s in concordantSpots], axis=1)
    concordantPbPbValues = np.stack([rng.normal(s.pbPbValue, s.pbPbStDev, stabilitySamples) for s in concordantSpots], axis=1)
    discordantUPbValues  = np.stack([rng.normal(s.uPbValue,  s.uPbStDev,  stabilitySamples) for s in discordantSpots], axis=1)
    discordantPbPbValues = np.stack([rng.normal(s.pbPbValue, s.pbPbStDev, stabilitySamples) for s in discordantSpots], axis=1)

    # ---- optional clustering of discordant ages ----
    full_labels = np.zeros(len(discordantSpots), dtype=int)
    if getattr(settings, "use_discordant_clustering", False):
        ages_y = np.asarray(settings.rimAges(), float)  # YEARS grid

        # Compute static proxies for **each discordant spot** (Ma)
        proxy_ma, keep_idx = [], []
        for idx, spot in enumerate(discordantSpots):
            proxy_y = lower_intercept_proxy(
                float(spot.uPbValue),
                float(spot.pbPbValue),
                ages_y,            # YEARS grid
            )
            if proxy_y is not None and np.isfinite(proxy_y):
                proxy_ma.append(proxy_y / 1e6)  # store in Ma
                keep_idx.append(idx)

        proxy_ma = np.asarray(proxy_ma, float)
        keep_idx = np.asarray(keep_idx, int)

        if proxy_ma.size >= 3:
            # Base GMM+BIC clustering on the static proxies
            core_labels, *_ = find_discordant_clusters(proxy_ma)
            min_pts, min_frac, sep_sig = _adaptive_gates(proxy_ma.size)

            if proxy_ma.size >= min_pts:
                soft = _soft_accept_labels(
                    core_labels, proxy_ma,
                    min_points=min_pts,
                    min_frac=min_frac,
                    sep_sig_thr=sep_sig,
                )
                if soft.max() >= 1:
                    full_labels = np.zeros(len(discordantSpots), dtype=int)
                    full_labels[keep_idx] = soft
                else:
                    full_labels = np.zeros(len(discordantSpots), dtype=int)
        else:
            full_labels = np.zeros(len(discordantSpots), dtype=int)

    for spot, lab in zip(discordantSpots, full_labels):
        spot.cluster_id = int(lab)

    # --------- sampling loop ---------
    per_run_times = []
    t0 = time.perf_counter()
    for j in range(stabilitySamples):
        if signals.halt():
            signals.cancelled()
            return False, "processing halted by user"

        t_run = time.perf_counter()

        labels_for_run = full_labels
        if getattr(settings, "use_discordant_clustering", False) and \
           getattr(settings, "relabel_clusters_per_run", False):
            # Per-run relabelling uses the SAME proxy recipe over the grid
            labels_for_run = _labels_from_this_run(
                discordantUPbValues[j], discordantPbPbValues[j], ages_y
            )

        run = MonteCarloRun(
            j, sample.name,
            concordantUPbValues[j],  concordantPbPbValues[j],
            discordantUPbValues[j],  discordantPbPbValues[j],
            discordant_labels=labels_for_run,
            settings=settings
        )
        _performSingleRun(settings, run)
        per_run_times.append(time.perf_counter() - t_run)

        progress = (j + 1) / stabilitySamples
        sample.addMonteCarloRun(run)
        signals.progress(ProgressType.SAMPLING, progress, sample.name, run)

    mc_elapsed = time.perf_counter() - t0
    try:
        grid_len = len(settings.rimAges())
    except Exception:
        grid_len = 0

    _write_runlog(dict(
        method="CDC", phase="MC",
        sample=sample.name, tier=_infer_tier(sample.name),
        R=stabilitySamples, n_grid=grid_len,
        elapsed_s=round(mc_elapsed, 3),
        per_run_median_s=round(float(np.median(per_run_times)), 4) if per_run_times else 0.0,
        per_run_p95_s=round(float(np.percentile(per_run_times, 95)), 4) if per_run_times else 0.0,
        rss_peak_mb=round(_rss_mb(), 1),
        python=platform.python_version(), numpy=np.__version__,
    ))

    _calculateOptimalAge(signals, sample, 1.0)
    return True, None

def _performRimAgeSampling_split_by_population(signals, sample):
    """
    Run CDC+ensemble separately for each concordant age population and
    merge the resulting peaks into a single catalogue.
    """
    settings = sample.calculationSettings
    setattr(settings, "timing_mode", TIMING_MODE)
    setattr(settings, "write_outputs", CDC_WRITE_OUTPUTS)

    eligibleSpots   = [s for s in sample.validSpots if not getattr(s, "reverseDiscordant", False)]
    concordantSpots = [s for s in eligibleSpots if s.concordant]
    discordantSpots = [s for s in eligibleSpots if not s.concordant]

    if not concordantSpots or not discordantSpots or len(discordantSpots) <= 2:
        return False, "insufficient spots for population split"

    # Cluster concordant ages into populations
    pop_labels_conc, n_pops, pop_means = _cluster_concordant_populations(concordantSpots, max_pops=3)
    pop_labels_disc = _assign_discordant_to_populations(discordantSpots, pop_means)

    all_runs = []
    all_peaks = []

    for pop_id in range(n_pops):
        sub_conc = [s for s, lab in zip(concordantSpots, pop_labels_conc) if lab == pop_id]
        sub_disc = [s for s, lab in zip(discordantSpots, pop_labels_disc) if lab == pop_id]
        if len(sub_conc) == 0 or len(sub_disc) < 3:
            continue

        # --- run the existing MC+ensemble logic on this subset ONLY ---
        ok, reason, runs_pop, peaks_pop = _run_cdc_on_subset(
            signals, f"{sample.name}_pop{pop_id}", settings, sub_conc, sub_disc
        )

        # tag peaks with population id
        for p in peaks_pop:
            p["population_id"] = pop_id
        all_runs.extend(runs_pop)
        all_peaks.extend(peaks_pop)

    # aggregate results back into the sample
    sample.monteCarloRuns = all_runs

    # Emit SAMPLING progress events for all runs so the UI sees them
    total_runs = len(all_runs)
    if total_runs:
        for j, run in enumerate(all_runs):
            progress = float(j + 1) / float(total_runs)
            signals.progress(ProgressType.SAMPLING, progress, sample.name, run)

    # Use the usual pipeline to build Goodness surfaces etc.
    _calculateOptimalAge(signals, sample, 1.0)

    # Overwrite catalogue with population-aware peaks (all_peaks)
    if all_peaks:
        sample.peak_catalogue = [
            dict(
                sample=sample.name,
                peak_no=i + 1,
                ci_low=p["ci_low"],
                age_ma=p["age_ma"],
                ci_high=p["ci_high"],
                support=p.get("support", float("nan")),
                population_id=p.get("population_id", None),
            )
            for i, p in enumerate(all_peaks)
        ]
        sample.summedKS_peaks_Ma = np.asarray([p["age_ma"] for p in all_peaks], float)
        sample.peak_uncertainty_str = fmt_peak_stats(
            [
                (p["age_ma"], p["ci_low"], p["ci_high"], p.get("support", float("nan")))
                for p in all_peaks
            ]
        )

    return True, None

def _run_cdc_on_subset(signals, sample_name: str, settings,
                       concordantSpots, discordantSpots):
    """
    Run the full MC + CDC + ensemble logic on a *subset* of spots:
    one concordant age population and its associated discordant grains.

    Parameters
    ----------
    signals : ProcessSignals-like
        Object providing .halt() and .progress(...) (used for cancellation).
    sample_name : str
        Name of the parent sample (used in seeding / diagnostics).
    settings : LeadLossCalculationSettings
        Calculation settings used for this subset (grid, MC runs, etc.).
    concordantSpots : list
        Subset of concordant spots assigned to this population.
    discordantSpots : list
        Subset of discordant spots assigned to this population.

    Returns
    -------
    ok : bool
        True if processing completed for this subset, False if skipped/aborted.
    reason : str or None
        Reason for skipping/aborting if ok is False, else None.
    runs_pop : list[MonteCarloRun]
        Monte Carlo runs for this subset.
    peaks_pop : list[dict]
        Ensemble peaks for this subset, each with keys {age_ma, ci_low, ci_high, support, ...}.
    """
    if not concordantSpots:
        return False, "no concordant spots in subset", [], []
    if len(discordantSpots) <= 2:
        return False, "fewer than 3 discordant spots in subset", [], []

    stabilitySamples = int(settings.monteCarloRuns)
    # Use a seed that depends on sample + subset to keep subset runs reproducible
    rng = np.random.default_rng(_seed_from_name(sample_name + "_pop"))

    # MC draws for this subset (R × N_conc / R × N_disc)
    concU  = np.stack([rng.normal(s.uPbValue,  s.uPbStDev,  stabilitySamples) for s in concordantSpots], axis=1)
    concPb = np.stack([rng.normal(s.pbPbValue, s.pbPbStDev, stabilitySamples) for s in concordantSpots], axis=1)
    discU  = np.stack([rng.normal(s.uPbValue,  s.uPbStDev,  stabilitySamples) for s in discordantSpots], axis=1)
    discPb = np.stack([rng.normal(s.pbPbValue, s.pbPbStDev, stabilitySamples) for s in discordantSpots], axis=1)

    # ---- optional LI clustering for this subset ----
    full_labels = np.zeros(len(discordantSpots), dtype=int)
    use_dc = bool(getattr(settings, "use_discordant_clustering", False))
    if use_dc:
        ages_y = np.asarray(settings.rimAges(), float)  # YEARS

        proxy_ma, keep_idx = [], []
        for idx, spot in enumerate(discordantSpots):
            proxy_y = lower_intercept_proxy(float(spot.uPbValue),
                                            float(spot.pbPbValue),
                                            ages_y)
            if proxy_y is not None and np.isfinite(proxy_y):
                proxy_ma.append(proxy_y / 1e6)  # store in Ma
                keep_idx.append(idx)

        proxy_ma = np.asarray(proxy_ma, float)
        keep_idx = np.asarray(keep_idx, int)

        if proxy_ma.size >= 3:
            core_labels, *_ = find_discordant_clusters(proxy_ma)
            min_pts, min_frac, sep_sig = _adaptive_gates(proxy_ma.size)
            if proxy_ma.size >= min_pts:
                soft = _soft_accept_labels(core_labels, proxy_ma,
                                           min_points=min_pts,
                                           min_frac=min_frac,
                                           sep_sig_thr=sep_sig)
                if soft.max() >= 1:
                    full_labels = np.zeros(len(discordantSpots), dtype=int)
                    full_labels[keep_idx] = soft

    runs_pop: List[MonteCarloRun] = []

    ages_y = np.asarray(settings.rimAges(), float)  # YEARS grid
    for j in range(stabilitySamples):
        if signals.halt():
            return False, "processing halted by user", [], []

        labels_for_run = full_labels
        if use_dc and bool(getattr(settings, "relabel_clusters_per_run", False)):
            labels_for_run = _labels_from_this_run(discU[j], discPb[j], ages_y)

        run = MonteCarloRun(
            j, sample_name,
            concU[j], concPb[j],
            discU[j], discPb[j],
            discordant_labels=labels_for_run,
            settings=settings
        )
        _performSingleRun(settings, run)
        runs_pop.append(run)

    # ---- Build peaks for this population from its runs ----
    if not runs_pop:
        return False, "no runs produced", [], []

    ages_ma = ages_y / 1e6
    S_runs_pen = _stack_min_across_clusters(runs_pop, ages_y, which='pen')
    smf = _smooth_frac_for_grid(ages_ma)
    Smed_pen, _, _ = robust_ensemble_curve(S_runs_pen, smooth_frac=smf)

    optima_ma_pop = np.array([r.optimal_pb_loss_age for r in runs_pop], float) / 1e6

    peaks_pop = build_ensemble_catalogue(
        sample_name, _infer_tier(sample_name), ages_ma, S_runs_pen,
        orientation='max', smooth_frac=smf,
        f_d=FD_DIST_FRAC, f_p=FP_PROM_FRAC, f_v=FV_VALLEY_FRAC, f_w=FW_WIN_FRAC,
        w_min_nodes=3, support_min=FS_SUPPORT, r_min=RMIN_RUNS, f_r=FR_RUN_REL,
        per_run_prom_frac=PER_RUN_PROM_FRAC, per_run_min_dist=PER_RUN_MIN_DIST,
        per_run_min_width=PER_RUN_MIN_WIDTH, per_run_require_full_prom=False,
        pen_ok_mask=None, cand_curve=Smed_pen,
        optima_ma=optima_ma_pop,          # <<< pass the subset optima here
    ) or []


    # Fallback: if no ensemble peak passes gates, promote the per-run optimal
    if not peaks_pop:
        opt_all = np.sort([r.optimal_pb_loss_age for r in runs_pop])
        if opt_all.size:
            optimalAge = float(np.median(opt_all))
            lower95    = float(opt_all[int(0.025 * len(opt_all))])
            upper95    = float(opt_all[int(np.ceil(0.975 * len(opt_all)) - 1)])
            peaks_pop = [dict(
                age_ma=optimalAge / 1e6,
                ci_low=lower95   / 1e6,
                ci_high=upper95  / 1e6,
                support=1.0,
                source="fallback_optimal",
            )]

    return True, None, runs_pop, peaks_pop

# ======================  Ensemble, KS, catalogue ======================

def _findOptimalIndex(valuesToCompare):
    minIndex, minValue = min(enumerate(valuesToCompare), key=lambda v: v[1])
    n = len(valuesToCompare)

    startMinIndex = minIndex
    while startMinIndex > 0 and valuesToCompare[startMinIndex - 1] == minValue:
        startMinIndex -= 1

    endMinIndex = minIndex
    while endMinIndex < n - 1 and valuesToCompare[endMinIndex + 1] == minValue:
        endMinIndex += 1

    if (endMinIndex != n - 1 and startMinIndex != 0) or (endMinIndex == n - 1 and startMinIndex == 0):
        return (endMinIndex + startMinIndex) // 2
    if startMinIndex == 0:
        return 0
    return n - 1

def _emit_summedKS(signals, sample, progress, ages_ma, y_curve, rows_for_ui):
    """
    Send the plotted curve together with peak positions and CI arrays, to BOTH:
      • sample.signals.summedKS (figure listens to this)
      • global signals.progress("summedKS", ...) (legacy bus)
    """
    ui_peaks_age = [float(r["age_ma"]) for r in rows_for_ui]
    ui_peaks_ci  = [[float(r["ci_low"]), float(r["ci_high"])] for r in rows_for_ui]
    ui_support   = [float(r.get("support", float("nan"))) for r in rows_for_ui]

    sample.summedKS_peaks_Ma   = np.asarray(ui_peaks_age, float)
    sample.summedKS_ci_low_Ma  = np.asarray([lo for lo, _ in ui_peaks_ci], float)
    sample.summedKS_ci_high_Ma = np.asarray([hi for _, hi in ui_peaks_ci], float)

    payload = (ages_ma.tolist(), y_curve.tolist(), ui_peaks_age, ui_peaks_ci, ui_support)
    try:
        if hasattr(sample.signals, "summedKS"):
            sample.signals.summedKS.emit(payload)
    except Exception:
        pass
    try:
        signals.progress("summedKS", progress, sample.name, payload)
    except TypeError:
        try:
            signals.progress("summedKS", progress, sample.name, (payload[0], payload[1], payload[2]))
        except Exception:
            pass

def _smooth_frac_for_grid(ages_ma):
    """Convert SMOOTH_MA (if >0) into a node fraction; else use SMOOTH_FRAC."""
    n = len(ages_ma)
    if n <= 1:
        return SMOOTH_FRAC
    if SMOOTH_MA > 0:
        step_ma = float(np.median(np.diff(ages_ma))) or 1e-9
        sigma_nodes = SMOOTH_MA / step_ma
        # robust_ensemble_curve expects sigma as a fraction of N
        return min(0.25, sigma_nodes / n)  # cap to avoid over-smoothing
    return SMOOTH_FRAC

def _collapse_ci_clusters(rows, width_mult: float = 1.0):
    """
    Collapse chains of peaks that are either:
      • CI-overlapping, OR
      • close in age compared to their widths.

    For each cluster, keep only the best-supported, narrowest peak
    and keep *its own* CI (no union widening).
    """
    if not rows or len(rows) <= 1:
        return rows

    # sort by age so we can scan left→right
    rows = sorted(rows, key=lambda r: float(r["age_ma"]))
    clusters = []
    current_cluster = [dict(rows[0])]

    def _same_cluster(a, b) -> bool:
        lo1, hi1 = float(a["ci_low"]),  float(a["ci_high"])
        lo2, hi2 = float(b["ci_low"]),  float(b["ci_high"])
        a1, a2   = float(a["age_ma"]),  float(b["age_ma"])

        # CI overlap?
        overlap = (lo2 <= hi1) and (hi2 >= lo1)

        # Age separation vs widths
        w1 = max(hi1 - lo1, 0.0)
        w2 = max(hi2 - lo2, 0.0)
        sep = abs(a2 - a1)

        # Treat as the same cluster if they overlap OR
        # separation is smaller than some multiple of the larger width.
        close = (w1 > 0.0 or w2 > 0.0) and sep <= width_mult * max(w1, w2)

        return overlap or close

    for r in rows[1:]:
        last = current_cluster[-1]
        if _same_cluster(last, r):
            current_cluster.append(dict(r))
        else:
            clusters.append(current_cluster)
            current_cluster = [dict(r)]
    clusters.append(current_cluster)

    collapsed = []
    for cl in clusters:
        # choose best-supported peak in the cluster; break ties with narrowest CI
        def _score(rr):
            sup   = float(rr.get("support", 0.0))
            width = float(rr["ci_high"]) - float(rr["ci_low"])
            return (sup, -width)  # higher support, then narrower

        best = max(cl, key=_score)
        collapsed.append(dict(best))  # copy

    # renumber
    for i, rr in enumerate(collapsed, 1):
        rr["peak_no"] = i

    return collapsed


def _calculateOptimalAge(signals, sample, progress):
    """
    FINAL, CONSISTENT PIPELINE
      • Build per-run goodness (RAW, PEN) as 1 - min_c D or 1 - min_c score.
      • Form robust-median ensemble curves Smed_raw and Smed_pen.
      • Build RAW catalogue from Smed_raw, PEN from Smed_pen.
      • Display **only** ensemble-derived peaks (no winners override).
      • Plot the SAME median curve used for candidates (alignment intact).
    """
    settings, runs = sample.calculationSettings, sample.monteCarloRuns
    if not runs:
        return

    # Grid
    ages_y  = np.asarray(settings.rimAges(), float)   # years
    ages_ma = ages_y / 1e6
    optima_ma = np.array([r.optimal_pb_loss_age for r in runs], float) / 1e6

    # Per-run goodness matrices (R × G)
    S_runs_raw = _stack_min_across_clusters(runs, ages_y, which='raw')  # 1 - KS D
    S_runs_pen = _stack_min_across_clusters(runs, ages_y, which='pen')  # 1 - score

    # Smoothing
    smf = _smooth_frac_for_grid(ages_ma)

    # Ensemble robust median curves + their dynamic ranges Δ (5–95 percentile span)
    Smed_raw, Delta_raw, _ = robust_ensemble_curve(S_runs_raw, smooth_frac=smf)
    Smed_pen, Delta_pen, _ = robust_ensemble_curve(S_runs_pen, smooth_frac=smf)

    # Decide which surface we will **plot** in the UI regardless of the toggle
    S_view = Smed_raw if (CATALOGUE_SURFACE == "RAW") else Smed_pen

    # Optimal-age stats across runs
    opt_all     = np.sort([r.optimal_pb_loss_age for r in runs])
    optimalAge  = float(np.median(opt_all))
    lower95     = opt_all[int(0.025 * len(opt_all))]
    upper95     = opt_all[int(np.ceil(0.975 * len(opt_all)) - 1)]
    stats       = [r.optimal_statistic for r in runs]
    meanD       = float(np.mean([s.test_statistics[0] for s in stats]))
    meanP       = float(np.mean([s.test_statistics[1] for s in stats]))
    meanInv     = float(np.mean([s.number_of_invalid_ages for s in stats]))
    meanSc      = float(np.mean([s.score                   for s in stats]))

    # ---- Toggle: build a catalogue only if enabled ----
    enabled = bool(getattr(settings, "enable_ensemble_peak_picking", False))
    if not enabled:
        # No catalogue: publish curve-only + empty peak list
        sample.peak_catalogue = []
        try:
            sample.signals.optimalAgeCalculated.emit()  # keeps summary panel in sync
        except Exception:
            pass
        _emit_summedKS(signals, sample, progress, ages_ma, S_view, rows_for_ui=[])
        signals.progress(
            ProgressType.OPTIMAL, 1.0, sample.name,
            (optimalAge, lower95, upper95, meanD, meanP, meanInv, meanSc, "—", [])
        )
        return
    # ------------------------------------------------------------------
    # Build catalogues strictly from the ensemble (no winners override).
    # If an ensemble surface is too flat (Δ < ENS_DELTA_MIN), we do not
    # attempt to pick peaks from it and rely on the optima fallback instead.
    # ------------------------------------------------------------------
    use_dc = bool(getattr(settings, "use_discordant_clustering", False))

    rows_raw: List[Dict] = []
    rows_pen: List[Dict] = []

    # Per-cluster ensembles (only if clustering enabled)
    if use_dc and USE_CLUSTER_CATALOGUE:
        S_by_cluster_raw = stack_goodness_by_cluster(runs, ages_y, which="raw")
        S_by_cluster_pen = stack_goodness_by_cluster(runs, ages_y, which="pen")

        for cid in sorted(S_by_cluster_raw.keys()):
            S_raw_k = np.asarray(S_by_cluster_raw[cid], float)
            S_pen_k = np.asarray(S_by_cluster_pen.get(cid, S_raw_k * np.nan), float)

            # Keep only runs that actually have this cluster (any finite entries)
            mask_runs = np.isfinite(S_raw_k).any(axis=1)
            if mask_runs.sum() < max(RMIN_RUNS, 2):
                continue

            S_raw_k = S_raw_k[mask_runs]
            S_pen_k = S_pen_k[mask_runs]
            optima_ma_k = optima_ma[mask_runs] 

            # Ensemble for this cluster
            Smed_raw_k, Delta_raw_k, _ = robust_ensemble_curve(S_raw_k, smooth_frac=smf)
            Smed_pen_k, Delta_pen_k, _ = robust_ensemble_curve(S_pen_k, smooth_frac=smf)

            # If both surfaces for this cluster are essentially flat or empty, skip it
            if (Smed_raw_k.size == 0 and Smed_pen_k.size == 0) or \
               (Delta_raw_k < ENS_DELTA_MIN and Delta_pen_k < ENS_DELTA_MIN):
                continue

            # RAW catalogue for this cluster (only if not too flat)
            if Smed_raw_k.size and Delta_raw_k >= ENS_DELTA_MIN:
                rows_raw_k = build_ensemble_catalogue(
                    sample.name, _infer_tier(sample.name), ages_ma, S_raw_k,
                    orientation="max", smooth_frac=smf,
                    f_d=FD_DIST_FRAC, f_p=FP_PROM_FRAC, f_v=FV_VALLEY_FRAC, f_w=FW_WIN_FRAC,
                    w_min_nodes=3, support_min=FS_SUPPORT, r_min=RMIN_RUNS, f_r=FR_RUN_REL,
                    per_run_prom_frac=PER_RUN_PROM_FRAC, per_run_min_dist=PER_RUN_MIN_DIST,
                    per_run_min_width=PER_RUN_MIN_WIDTH, per_run_require_full_prom=False,
                    pen_ok_mask=None, cand_curve=Smed_raw_k, height_frac=FH_HEIGHT_FRAC, optima_ma=optima_ma_k,
                ) or []
                for r in rows_raw_k:
                    r["cluster_id"] = int(cid)
                rows_raw.extend(rows_raw_k)

            # PEN catalogue for this cluster (only if not too flat)
            if Smed_pen_k.size and Delta_pen_k >= ENS_DELTA_MIN:
                rows_pen_k = build_ensemble_catalogue(
                    sample.name, _infer_tier(sample.name), ages_ma, S_pen_k,
                    orientation="max", smooth_frac=smf,
                    f_d=FD_DIST_FRAC, f_p=FP_PROM_FRAC, f_v=FV_VALLEY_FRAC, f_w=FW_WIN_FRAC,
                    w_min_nodes=3, support_min=FS_SUPPORT, r_min=RMIN_RUNS, f_r=FR_RUN_REL,
                    per_run_prom_frac=PER_RUN_PROM_FRAC, per_run_min_dist=PER_RUN_MIN_DIST,
                    per_run_min_width=PER_RUN_MIN_WIDTH, per_run_require_full_prom=False,
                    pen_ok_mask=None, cand_curve=Smed_pen_k, height_frac=FH_HEIGHT_FRAC, optima_ma=optima_ma_k,
                ) or []
                for r in rows_pen_k:
                    r["cluster_id"] = int(cid)
                rows_pen.extend(rows_pen_k)

    # Global RAW ensemble (only if not flat and nothing robust from clusters)
    if not rows_raw and Delta_raw >= ENS_DELTA_MIN:
        rows_raw = build_ensemble_catalogue(
            sample.name, _infer_tier(sample.name), ages_ma, S_runs_raw,
            orientation="max", smooth_frac=smf,
            f_d=FD_DIST_FRAC, f_p=FP_PROM_FRAC, f_v=FV_VALLEY_FRAC, f_w=FW_WIN_FRAC,
            w_min_nodes=3, support_min=FS_SUPPORT, r_min=RMIN_RUNS, f_r=FR_RUN_REL,
            per_run_prom_frac=PER_RUN_PROM_FRAC, per_run_min_dist=PER_RUN_MIN_DIST,
            per_run_min_width=PER_RUN_MIN_WIDTH, per_run_require_full_prom=False,
            pen_ok_mask=None, cand_curve=Smed_raw, height_frac=FH_HEIGHT_FRAC, optima_ma=optima_ma,
        ) or []

        if not rows_raw and Delta_raw >= ENS_DELTA_MIN:
            rows_raw = build_ensemble_catalogue(
                sample.name, _infer_tier(sample.name), ages_ma, S_runs_raw,
                orientation="max", smooth_frac=smf,
                f_d=FD_DIST_FRAC, f_p=max(0.5 * FP_PROM_FRAC, 0.01),
                f_v=FV_VALLEY_FRAC, f_w=FW_WIN_FRAC,
                w_min_nodes=3, support_min=FS_SUPPORT, r_min=RMIN_RUNS, f_r=FR_RUN_REL,
                per_run_prom_frac=PER_RUN_PROM_FRAC, per_run_min_dist=PER_RUN_MIN_DIST,
                per_run_min_width=PER_RUN_MIN_WIDTH, per_run_require_full_prom=False,
                pen_ok_mask=None, cand_curve=Smed_raw, height_frac=FH_HEIGHT_FRAC, optima_ma=optima_ma,
            ) or []

    # Global PEN ensemble (only if not flat)
    if not rows_pen and Delta_pen >= ENS_DELTA_MIN:
        rows_pen = build_ensemble_catalogue(
            sample.name, _infer_tier(sample.name), ages_ma, S_runs_pen,
            orientation="max", smooth_frac=smf,
            f_d=FD_DIST_FRAC, f_p=FP_PROM_FRAC, f_v=FV_VALLEY_FRAC, f_w=FW_WIN_FRAC,
            w_min_nodes=3, support_min=FS_SUPPORT, r_min=RMIN_RUNS, f_r=FR_RUN_REL,
            per_run_prom_frac=PER_RUN_PROM_FRAC, per_run_min_dist=PER_RUN_MIN_DIST,
            per_run_min_width=PER_RUN_MIN_WIDTH, per_run_require_full_prom=False,
            pen_ok_mask=None, cand_curve=Smed_pen, height_frac=FH_HEIGHT_FRAC, optima_ma=optima_ma,
        ) or []

    # Fallback:
    # Only promote the median of the per-run optima to a pseudo-peak if the
    # ensemble surface has some structure (Δ >= ENS_DELTA_MIN). If both RAW
    # and PEN surfaces are essentially flat, leave the catalogue empty.
    if (not rows_raw) and (not rows_pen) and len(opt_all) > 0:


        # Fallback: if no ensemble peaks survive, **do not** invent one.
        # Leave the catalogue empty – the single CDC optimal age is still
        # reported in the summary, but there is no robust ensemble peak.        
        rows_raw = []
        rows_pen = []


    # Choose which to DISPLAY in the UI strictly from the ensemble surface.
    ui_surface = CATALOGUE_SURFACE  # "RAW" or "PEN"
    if (ui_surface == "RAW") and rows_raw:
        rows_for_ui = rows_raw
        S_view = Smed_raw
    elif rows_pen:
        rows_for_ui = rows_pen
        S_view = Smed_pen
    else:
        rows_for_ui = []
        S_view = Smed_pen if ui_surface != "RAW" else Smed_raw

    _ensure_output_dirs()

    rows_for_ui = _collapse_ci_clusters(rows_for_ui)
    rows_raw    = _collapse_ci_clusters(rows_raw)
    rows_pen    = _collapse_ci_clusters(rows_pen)

    # Ensure snapped age lies inside its CI (preserve width >= 1 step)
    if rows_for_ui:
        step = float(np.median(np.diff(ages_ma))) if ages_ma.size >= 2 else 5.0
        min_age, max_age = float(ages_ma[0]), float(ages_ma[-1])
        fixed = []
        for r in rows_for_ui:
            a  = float(r["age_ma"])
            lo = float(r["ci_low"])
            hi = float(r["ci_high"])
            w  = max(hi - lo, step)

            if (a < lo) or (a > hi):
                lo, hi = a - 0.5 * w, a + 0.5 * w
                lo, hi = max(lo, min_age), min(hi, max_age)
                if (hi - lo) < step:
                    lo, hi = max(a - step, min_age), min(a + step, max_age)

            fixed.append(dict(r, ci_low=lo, ci_high=hi))
        rows_for_ui = fixed

    # Enforce a minimum CI width and drop boundary‑degenerate peaks
    if rows_for_ui:
        step = float(np.median(np.diff(ages_ma))) if ages_ma.size >= 2 else 5.0
        min_age, max_age = float(ages_ma[0]), float(ages_ma[-1])
        cleaned = []
        for r in rows_for_ui:
            a  = float(r["age_ma"])
            lo = float(r["ci_low"])
            hi = float(r["ci_high"])
            if (hi - lo) < step:
                lo, hi = a - step, a + step
            near_edge  = (a - min_age) <= step or (max_age - a) <= step
            degenerate = (hi - lo) <= 0.75 * step
            if near_edge and degenerate:
                if float(r.get("support", 0.0)) >= max(FS_SUPPORT, 0.12):
                    lo, hi = a - step, a + step
                else:
                    continue
            cleaned.append(dict(r, ci_low=lo, ci_high=hi))
        rows_for_ui = cleaned

    for r in rows_for_ui:
        a = float(r["age_ma"])
        if not (float(r["ci_low"]) <= a <= float(r["ci_high"])):
            r["ci_low"]  = min(float(r["ci_low"]),  a)
            r["ci_high"] = max(float(r["ci_high"]), a)

    if CDC_WRITE_OUTPUTS:
        _append_catalogue_rows(sample.name, rows_pen, dest_path=CATALOGUE_CSV_PEN)
        _append_catalogue_rows(sample.name, rows_raw, dest_path=CATALOGUE_CSV_RAW)
        prefix = _safe_prefix(sample.name)
        np.savez_compressed(
            DIAG_DIR / f"{prefix}_runs_S.npz",
            age_Ma=ages_ma, S_runs_raw=S_runs_raw, S_runs_pen=S_runs_pen,
            optima_Ma=np.array([r.optimal_pb_loss_age / 1e6 for r in runs], float),
        )
        np.savez_compressed(
            DIAG_DIR / f"{prefix}_ensemble_surfaces.npz",
            age_Ma=ages_ma, Smed_raw=Smed_raw, Smed_pen=Smed_pen, S_view=S_view,
            peaks_age_Ma=np.array([r["age_ma"] for r in rows_for_ui], float),
            peaks_ci_low=np.array([r["ci_low"] for r in rows_for_ui], float),
            peaks_ci_high=np.array([r["ci_high"] for r in rows_for_ui], float),
            peaks_support=np.array([r["support"] for r in rows_for_ui], float),
        )
        for r_idx, r in enumerate(runs, start=1):
            d_raw = np.array([r.statistics_by_pb_loss_age[a].test_statistics[0] for a in ages_y], float)
            d_pen = np.array([r.statistics_by_pb_loss_age[a].score for a in ages_y], float)
            np.savez_compressed(
                DIAG_DIR / f"{prefix}_{r_idx:03d}.npz",
                age_Ma=ages_ma, D_raw=d_raw, D_pen=d_pen,
                S_raw=1.0 - d_raw, S_pen=1.0 - d_pen,
                opt_Ma=float(r.optimal_pb_loss_age / 1e6),
            )

    # ---- Drop peaks with absurdly wide CIs (essentially the entire grid) ----
    if rows_for_ui:
        total_span = float(ages_ma[-1] - ages_ma[0])
        MAX_CI_FRAC = 0.5  # e.g. drop peaks whose CI spans >70% of the window

        filtered = []
        for r in rows_for_ui:
            width = float(r["ci_high"] - r["ci_low"])
            if width > MAX_CI_FRAC * total_span:
                # This “peak” is so broad that it’s effectively unconstrained → discard
                continue
            filtered.append(r)
        rows_for_ui = filtered

        # Keep rows_raw/rows_pen in sync with what we actually show
        def _keep_same(rows, keep):
            if not rows or not keep:
                return []
            keep_ages = {float(r["age_ma"]) for r in keep}
            return [r for r in rows if float(r.get("age_ma", float("nan"))) in keep_ages]

        rows_raw = _keep_same(rows_raw, rows_for_ui)
        rows_pen = _keep_same(rows_pen, rows_for_ui)

    # Publish to UI
    catalogue = [(r["age_ma"], r["ci_low"], r["ci_high"], r["support"]) for r in rows_for_ui]
    red_peaks = np.asarray([m for m, *_ in catalogue], float)
    peak_str  = fmt_peak_stats(catalogue) if catalogue else "—"

    sample.summedKS_peaks_Ma = red_peaks
    sample.peak_uncertainty_str = peak_str
    sample.peak_catalogue = [
        dict(sample=sample.name, peak_no=i + 1, ci_low=lo, age_ma=med, ci_high=hi, support=sup)
        for i, (med, lo, hi, sup) in enumerate(catalogue)
    ]

    try:
        sample.signals.optimalAgeCalculated.emit()
    except Exception:
        pass

    _emit_summedKS(signals, sample, progress, ages_ma, S_view, rows_for_ui)

    signals.progress(
        ProgressType.OPTIMAL, 1.0, sample.name,
        (optimalAge, lower95, upper95, meanD, meanP, meanInv, meanSc, peak_str, catalogue)
    )

def calculateHeatmapData(signals, runs, settings):
    resolution = config.HEATMAP_RESOLUTION

    colData = [[] for _ in range(resolution)]
    for run in runs:
        if run is None:
            continue
        for col in range(resolution):
            colData[col].append(run.heatmapColumnData[col])

    cache = {}
    data = [[0 for _ in range(resolution)] for _ in range(resolution)]
    for col in range(resolution):
        if len(colData[col]) == 0:
            continue

        mean = np.mean(colData[col])
        stdDev = np.std(colData[col])
        if stdDev < 10 ** -7:
            stdDev = 0

        if (mean, stdDev) not in cache:
            if stdDev == 0:
                meanRow = (resolution - 1) if mean == 1.0 else int(mean * resolution)
                result = [1 if i >= meanRow else 0 for i in range(resolution + 1)]
            else:
                rv = sp.stats.norm(mean, stdDev)
                result = rv.cdf(np.linspace(0, 1, resolution + 1))
            cache[(mean, stdDev)] = result

        cdfs = cache[(mean, stdDev)]
        for row in range(resolution):
            data[row][col] = cdfs[row + 1] - cdfs[row]

    signals.progress(data, settings)
