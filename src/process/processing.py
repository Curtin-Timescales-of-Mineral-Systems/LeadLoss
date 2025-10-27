# process/processing.py
import time
from enum import Enum
import numpy as np
import scipy as sp

from model.monteCarloRun import MonteCarloRun
from process import calculations
from model.settings.calculation import DiscordanceClassificationMethod
from utils import config
from process.discordantClustering import find_discordant_clusters, _labels_from_this_run, _stack_min_across_clusters, lower_intercept_proxy, _soft_accept_labels
from utils.peakHelpers import fmt_peak_stats
from process.ensemble import robust_ensemble_curve, build_ensemble_catalogue
from pathlib import Path
import csv, sys, platform, resource, os, re, zlib

# ======================  EXPOSED KNOBS (keep your GUI)  ======================
SMOOTH_MA   = float(os.environ.get("CDC_SMOOTH_MA", "0.0"))
SMOOTH_FRAC = float(os.environ.get("CDC_SMOOTH_FRAC", "0.01"))  # used if SMOOTH_MA <= 0

# Per-run detector gates – restore old
PER_RUN_PROM_FRAC = 0.06
PER_RUN_MIN_DIST  = 3
PER_RUN_MIN_WIDTH = 3

# Ensemble gates – restore old “conservative” thresholds
FD_DIST_FRAC   = 0.05
FP_PROM_FRAC   = 0.06
FW_WIN_FRAC    = 0.08
FR_RUN_REL     = 0.35
FS_SUPPORT     = 0.10
RMIN_RUNS      = 3
FV_VALLEY_FRAC = 0.10   # older shoulder-merge depth

# Optional diagnostics / CSV outputs
DIAG_DIR = Path("/Users/lucymathieson/Desktop/Desktop - Lucy’s MacBook Pro - 1/LeadLoss-2/diag_ks")
RUNLOG   = Path("/Users/lucymathieson/Desktop/Peak-Picking-Manuscript-Python/runtime_log.csv")

_BASE_CATALOGUE   = Path("/Users/lucymathieson/Desktop/Peak-Picking-Manuscript-Python/ensemble_catalogue")
CATALOGUE_CSV_PEN = _BASE_CATALOGUE.with_suffix(".csv")                     # penalised catalogue
CATALOGUE_CSV_RAW = _BASE_CATALOGUE.with_name(_BASE_CATALOGUE.name + "_np").with_suffix(".csv")  # raw / non-penalised

# Which catalogue to show in UI by default (no effect on detection)
UI_SURFACE        = os.environ.get("CDC_UI_SURFACE", "PEN").strip().upper()
CATALOGUE_SURFACE = os.environ.get("CDC_CATALOGUE_SURFACE", "PEN").strip().upper()

if UI_SURFACE not in {"RAW", "PEN"}:
    UI_SURFACE = "PEN"
if CATALOGUE_SURFACE not in {"RAW", "PEN"}:
    CATALOGUE_SURFACE = UI_SURFACE

CDC_WRITE_OUTPUTS = bool(int(os.environ.get("CDC_WRITE_OUTPUTS", "0")))
CDC_ENABLE_RUNLOG = bool(int(os.environ.get("CDC_ENABLE_RUNLOG", "0")))

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
    if not CDC_WRITE_OUTPUTS:
        return
    CATALOGUE_CSV_PEN.parent.mkdir(parents=True, exist_ok=True)
    CATALOGUE_CSV_RAW.parent.mkdir(parents=True, exist_ok=True)
    RUNLOG.parent.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

if CDC_WRITE_OUTPUTS:
    _ensure_output_dirs()

RUN_FIELDS = [
    "method","phase","sample","tier","R","n_grid","elapsed_s",
    "per_run_median_s","per_run_p95_s","rss_peak_mb","python","numpy"
]

# --- add this helper (exactly as you had before) ---
def _normalize_runs(S_runs):
    V = np.array(S_runs, float, copy=True)
    for r in range(V.shape[0]):
        row = V[r]; m = np.isfinite(row)
        if not m.any():
            continue
        p5, p95 = np.nanpercentile(row[m], [5, 95])
        if np.isfinite(p5) and np.isfinite(p95) and p95 > p5:
            row[m] = np.clip((row[m] - p5) / (p95 - p5), 0.0, 1.0)
        else:
            row[m] = 0.0
        V[r] = row
    return V

def _safe_prefix(name: str) -> str:
    s = str(name or "").strip()
    s = re.sub(r"[^\w\-.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_.")
    return s or "sample"

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

def _infer_tier(sample_name: str) -> str:
    s = (sample_name or "").strip()
    return s[-1].upper() if s and s[-1].isalpha() else ""

def _emit_summedKS(signals, sample, progress, ages_ma, y_curve, rows_for_ui):
    """
    Send the plotted curve together with peak positions and CI arrays, to BOTH:
      • sample.signals.summedKS (figure listens to this)
      • global signals.progress("summedKS", ...) (legacy bus)
    """
    ui_peaks_age = [float(r["age_ma"]) for r in rows_for_ui]
    ui_peaks_ci  = [[float(r["ci_low"]), float(r["ci_high"])] for r in rows_for_ui]
    ui_support   = [float(r.get("support", float("nan"))) for r in rows_for_ui]

    import numpy as _np
    sample.summedKS_peaks_Ma   = _np.asarray(ui_peaks_age, float)
    sample.summedKS_ci_low_Ma  = _np.asarray([lo for lo, _ in ui_peaks_ci], float)
    sample.summedKS_ci_high_Ma = _np.asarray([hi for _, hi in ui_peaks_ci], float)

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
        completed, skip_reason = _calculateConcordantAges(signals, sample)
        if not completed:
            return False, skip_reason

        completed, skip_reason = _performRimAgeSampling(signals, sample)
        if not completed:
            return False, skip_reason

        return True, None
    finally:
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
    sampleNameText = f" for '{sample.name}'" if sample.name else ""
    signals.newTask("Classifying points" + sampleNameText + "...")

    settings = sample.calculationSettings
    n_spots  = max(1, len(sample.validSpots))
    timePerRow = TIME_PER_TASK / n_spots

    concordancy = []
    discordances = []
    for i, spot in enumerate(sample.validSpots):
        signals.progress(ProgressType.CONCORDANCE, i / n_spots)

        time.sleep(timePerRow)
        if signals.halt():
            signals.cancelled()
            return False, "processing halted by user"

        if settings.discordanceClassificationMethod == DiscordanceClassificationMethod.PERCENTAGE:
            discordance = calculations.discordance(spot.uPbValue, spot.pbPbValue)
            concordant  = discordance < settings.discordancePercentageCutoff
        else:
            discordance = None
            concordant = calculations.isConcordantErrorEllipse(
                spot.uPbValue,  spot.uPbStDev,
                spot.pbPbValue, spot.pbPbStDev,
                settings.discordanceEllipseSigmas
            )

        discordances.append(discordance)
        concordancy.append(concordant)

    sample.updateConcordance(concordancy, discordances)
    signals.progress(ProgressType.CONCORDANCE, 1.0, sample.name, concordancy, discordances)
    return True, None

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

def _performRimAgeSampling(signals, sample):
    sample.monteCarloRuns = []
    sample.peak_catalogue = []
    sampleNameText = f" for '{sample.name}'" if sample.name else ""
    signals.newTask("Sampling Pb-loss age distributions" + sampleNameText + "...")

    settings = sample.calculationSettings
    setattr(settings, "timing_mode", TIMING_MODE)
    setattr(settings, "write_outputs", CDC_WRITE_OUTPUTS)


    concordantSpots = [s for s in sample.validSpots if s.concordant]
    discordantSpots = [s for s in sample.validSpots if not s.concordant]

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

        proxy_ma, keep_idx = [], []
        for idx, s in enumerate(discordantSpots):
            proxy_y = lower_intercept_proxy(s.uPbValue, s.pbPbValue, ages_y)
            if proxy_y is not None and np.isfinite(proxy_y):
                proxy_ma.append(proxy_y / 1e6)  # cluster in Ma
                keep_idx.append(idx)

        proxy_ma = np.asarray(proxy_ma, float)
        keep_idx = np.asarray(keep_idx, int)

        min_pts  = int(os.environ.get("CDC_MIN_POINTS_GMM", "10"))
        if proxy_ma.size >= min_pts:
            core_labels, *_ = find_discordant_clusters(proxy_ma)
            # Use the same soft-accept logic as relabelling (size + separation)
            soft = _soft_accept_labels(core_labels, proxy_ma,
                                       min_points=min_pts,
                                       min_frac=float(os.environ.get("CDC_GMM_MIN_FRAC", "0.10")),
                                       sep_sig_thr=float(os.environ.get("CDC_GMM_SEP_SIG", "1.2")))
            if soft.max() >= 1:
                full_labels = np.zeros(len(discordantSpots), dtype=int)
                full_labels[keep_idx] = soft
            else:
                full_labels = np.zeros(len(discordantSpots), dtype=int)
        else:
            full_labels = np.zeros(len(discordantSpots), dtype=int)

        # Optional debug
        if os.environ.get("CDC_DEBUG_CLUSTERS") == "1":
            import collections
            cnt = collections.Counter(full_labels.tolist())
            print(f"[DC] proxy range (Ma): "
                  f"{np.nanmin(proxy_ma) if proxy_ma.size else np.nan}–"
                  f"{np.nanmax(proxy_ma) if proxy_ma.size else np.nan} ; "
                  f"labels: {dict(cnt)}")

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

    if getattr(settings, "use_discordant_clustering", False):
        uniq, cnt = np.unique(full_labels, return_counts=True)
        print(f"[DC] base labels: {dict(zip(uniq.tolist(), cnt.tolist()))}")
        if getattr(settings, "relabel_clusters_per_run", False):
            uniqR, cntR = np.unique(labels_for_run, return_counts=True)
            print(f"[DC] relabel@run: {dict(zip(uniqR.tolist(), cntR.tolist()))}")

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

def _performSingleRun(settings, run):
    for age in settings.rimAges():
        run.samplePbLossAge(age, settings.dissimilarityTest, settings.penaliseInvalidAges)
    run.calculateOptimalAge()
    run.createHeatmapData(settings.minimumRimAge, settings.maximumRimAge, config.HEATMAP_RESOLUTION)

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

def _calculateOptimalAge(signals, sample, progress):
    """
    FINAL, CONSISTENT PIPELINE
      • Build per-run goodness (RAW, PEN) as 1 - min_c D or 1 - min_c score.
      • Form robust-median ensemble curves Smed_raw and Smed_pen.
      • Build RAW catalogue from Smed_raw, PEN from Smed_pen.
      • Choose which catalogue to display based on CATALOGUE_SURFACE.
      • Plot the SAME median curve used for candidates (alignment intact).
    """
    settings, runs = sample.calculationSettings, sample.monteCarloRuns
    if not runs:
        return

    # ---------- helper ----------
    def _snap_to_local_max(rows, ages, y_curve, win_ma=50.0):
        if not rows:
            return rows
        from scipy.signal import find_peaks
        ages = np.asarray(ages, float)
        y_curve = np.asarray(y_curve, float)
        step = float(np.median(np.diff(ages))) or 10.0
        w = int(max(2, round(win_ma / step)))
        snapped = []
        for r in rows:
            j0 = int(np.argmin(np.abs(ages - r["age_ma"])))
            lo, hi = max(1, j0 - w), min(len(ages) - 2, j0 + w)
            loc, _ = find_peaks(y_curve[lo:hi+1], distance=1)
            j = lo + int(loc[np.argmax(y_curve[lo:hi+1][loc])]) if loc.size else j0
            snapped.append(dict(r, age_ma=float(ages[j])))
        return snapped
    # ----------------------------

    # Grid
    ages_y  = np.asarray(settings.rimAges(), float)   # years
    ages_ma = ages_y / 1e6

    # Per-run goodness matrices (R × G)
    S_runs_raw = _stack_min_across_clusters(runs, ages_y, which='raw')  # 1 - KS D
    S_runs_pen = _stack_min_across_clusters(runs, ages_y, which='pen')  # 1 - score

    # Smoothing
    smf = _smooth_frac_for_grid(ages_ma)

    # Ensemble robust median curves
    Smed_raw, _, _ = robust_ensemble_curve(S_runs_raw, smooth_frac=smf)
    Smed_pen, _, _ = robust_ensemble_curve(S_runs_pen, smooth_frac=smf)

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

    # Build catalogues
    rows_raw = build_ensemble_catalogue(
        sample.name, _infer_tier(sample.name), ages_ma, S_runs_raw,
        orientation='max', smooth_frac=smf,
        f_d=FD_DIST_FRAC, f_p=FP_PROM_FRAC, f_v=FV_VALLEY_FRAC, f_w=FW_WIN_FRAC,
        w_min_nodes=1, support_min=FS_SUPPORT, r_min=RMIN_RUNS, f_r=FR_RUN_REL,
        per_run_prom_frac=PER_RUN_PROM_FRAC, per_run_min_dist=PER_RUN_MIN_DIST,
        per_run_min_width=PER_RUN_MIN_WIDTH, per_run_require_full_prom=False,
        pen_ok_mask=None, cand_curve=Smed_raw
    ) or []

    if not rows_raw:
        rows_raw = build_ensemble_catalogue(
            sample.name, _infer_tier(sample.name), ages_ma, S_runs_raw,
            orientation='max', smooth_frac=smf,
            f_d=FD_DIST_FRAC, f_p=max(0.5 * FP_PROM_FRAC, 0.01), f_v=FV_VALLEY_FRAC, f_w=FW_WIN_FRAC,
            w_min_nodes=1, support_min=FS_SUPPORT, r_min=RMIN_RUNS, f_r=FR_RUN_REL,
            per_run_prom_frac=PER_RUN_PROM_FRAC, per_run_min_dist=PER_RUN_MIN_DIST,
            per_run_min_width=PER_RUN_MIN_WIDTH, per_run_require_full_prom=False,
            pen_ok_mask=None, cand_curve=Smed_raw
        ) or []

    rows_pen = build_ensemble_catalogue(
        sample.name, _infer_tier(sample.name), ages_ma, S_runs_pen,
        orientation='max', smooth_frac=smf,
        f_d=FD_DIST_FRAC, f_p=FP_PROM_FRAC, f_v=FV_VALLEY_FRAC, f_w=FW_WIN_FRAC,
        w_min_nodes=1, support_min=FS_SUPPORT, r_min=RMIN_RUNS, f_r=FR_RUN_REL,
        per_run_prom_frac=PER_RUN_PROM_FRAC, per_run_min_dist=PER_RUN_MIN_DIST,
        per_run_min_width=PER_RUN_MIN_WIDTH, per_run_require_full_prom=False,
        pen_ok_mask=None, cand_curve=Smed_pen
    ) or []

    _ensure_output_dirs()

    # Choose which to DISPLAY (original behavior): by CATALOGUE_SURFACE
    ui_surface = CATALOGUE_SURFACE  # "RAW" or "PEN"
    if (ui_surface == "RAW") and rows_raw:
        rows_for_ui = rows_raw
        S_view = Smed_raw
    elif rows_pen:
        rows_for_ui = rows_pen
        S_view = Smed_pen
    else:
        rows_for_ui = rows_raw
        S_view = Smed_raw

    # Snap markers to the crest of the curve actually being drawn
    rows_for_ui = _snap_to_local_max(rows_for_ui, ages_ma, S_view)
    rows_for_ui = _norm_catalogue(rows_for_ui)

    # Optional diagnostics
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

    # Publish to UI
    catalogue = [(r["age_ma"], r["ci_low"], r["ci_high"], r["support"]) for r in rows_for_ui]
    red_peaks = np.asarray([m for m, *_ in catalogue], float)
    peak_str  = fmt_peak_stats(catalogue) if catalogue else "—"

    sample.summedKS_peaks_Ma = red_peaks
    sample.peak_uncertainty_str = peak_str
    sample.peak_catalogue = [
        dict(sample=sample.name, peak_no=i+1, ci_low=lo, age_ma=med, ci_high=hi, support=sup)
        for i, (med, lo, hi, sup) in enumerate(catalogue)
    ]

    try:
        sample.signals.optimalAgeCalculated.emit()
    except Exception:
        pass

    _emit_summedKS(signals, sample, progress, ages_ma, S_view, rows_for_ui)
    print("[CDC] CATALOGUE_SURFACE:", ui_surface, "S_view:", "RAW" if S_view is Smed_raw else "PEN")

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
