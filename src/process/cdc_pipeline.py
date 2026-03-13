"""CDC processing pipeline.

A refactor of the original `process/processing.py`:
- configuration lives in `process.cdc_config`
- diagnostics/paper exports live in `process.cdc_diagnostics`
- Tera-W maths lives in `process.cdc_tw`

The GUI should continue to call `process.processing.processSamples(...)`.
"""

from __future__ import annotations

import platform
import time
from enum import Enum
from typing import Dict, List

import numpy as np

from model.monteCarloRun import MonteCarloRun
from model.settings.calculation import DiscordanceClassificationMethod
from process import calculations
from process.ensemble import robust_ensemble_curve, build_ensemble_catalogue
from process.cdc_config import (
    CATALOGUE_CSV_PEN,
    CATALOGUE_CSV_RAW,
    CATALOGUE_SURFACE,
    CDC_WRITE_OUTPUTS,
    ENS_DELTA_MIN,
    FD_DIST_FRAC,
    FH_HEIGHT_FRAC,
    FP_PROM_FRAC,
    FR_RUN_REL,
    FS_SUPPORT,
    FW_WIN_FRAC,
    KS_EXPORT_ROOT,
    MERGE_NEARBY_PEAKS,
    MONO_DY_EPS_FRAC,
    MONO_MAX_TURNS,
    PLATEAU_DEDUPE,
    PLATEAU_DEDUPE_MIN_OVERLAP_FRAC,
    PLATEAU_DEDUPE_RADIUS_STEPS,
    PER_RUN_MIN_DIST,
    PER_RUN_MIN_WIDTH,
    PER_RUN_PROM_FRAC,
    RMIN_RUNS,
    FV_VALLEY_FRAC,
    RUNLOG,
    SMOOTH_MA,
    SMOOTH_FRAC,
    TIMING_MODE,
)
from process.cdc_diagnostics import (
    append_catalogue_rows as _append_catalogue_rows,
    ensure_output_dirs as _ensure_output_dirs,
    export_legacy_ks as _export_legacy_ks,
    reset_csv as _reset_csv,
    rss_mb as _rss_mb,
    write_npz_diagnostics as _write_npz_diagnostics,
    write_runlog as _write_runlog,
)
from process.cdc_tw import is_reverse_discordant as _is_reverse_discordant
from process.cdc_utils import infer_tier as _infer_tier, seed_from_name as _seed_from_name
from utils import config
from utils.peakHelpers import fmt_peak_stats

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
        
        st = sample.calculationSettings

        # 3) Edge-guard: widen once if many per-run optima hug a boundary
        try:
            ages_ma = np.asarray(st.rimAges(), float) / 1e6
            if ages_ma.size >= 2 and sample.monteCarloRuns:
                # robust step estimate
                raw_step = float(np.median(np.diff(ages_ma))) if ages_ma.size >= 2 else 0.0
                if not np.isfinite(raw_step) or raw_step <= 0.0:
                    raw_step = 5.0
                step_ma = raw_step

                # Edge-guard follows the same primary channel used for the run.
                prefer_pen = bool(getattr(st, "penaliseInvalidAges", False))
                if prefer_pen:
                    opt_ma = np.array(
                        [r.optimal_pb_loss_age for r in sample.monteCarloRuns],
                        dtype=float,
                    ) / 1e6
                else:
                    opt_ma = np.array(
                        [_raw_optimum_age_ma(r) for r in sample.monteCarloRuns],
                        dtype=float,
                    )
                opt_ma  = opt_ma[np.isfinite(opt_ma)]
                if opt_ma.size:
                    hit_lo    = np.mean(opt_ma <= (ages_ma[0] + step_ma))      # young boundary
                    hit_hi    = np.mean(opt_ma >= (ages_ma[-1] - step_ma))     # old boundary
                    edge_hits = max(hit_lo, hit_hi)

                    if edge_hits > 0.20:
                        # Keep the search window strictly > 0 Ma; 0 can trigger
                        # singular behaviour in TW transforms used downstream.
                        MIN_MODEL_AGE_Y = 1.0e6
                        span_y        = float(st.maximumRimAge) - float(st.minimumRimAge)
                        if np.isfinite(span_y) and span_y > 0.0:
                            expand_y      = 0.20 * span_y
                            widen_younger = (hit_lo >= hit_hi)

                            if widen_younger:
                                new_min = max(MIN_MODEL_AGE_Y, float(st.minimumRimAge) - expand_y)
                                new_max = float(st.maximumRimAge)
                            else:
                                new_min = float(st.minimumRimAge)
                                new_max = float(st.maximumRimAge) + expand_y

                            if np.isfinite(new_min) and np.isfinite(new_max) and new_max > new_min:
                                # keep grid step roughly constant
                                ages_ma_new = np.asarray(st.rimAges(), float) / 1e6
                                raw_step2 = float(np.median(np.diff(ages_ma_new))) if ages_ma_new.size >= 2 else 0.0
                                if not np.isfinite(raw_step2) or raw_step2 <= 0.0:
                                    raw_step2 = step_ma or 5.0
                                step_ma2 = raw_step2

                                span_ma_new = (new_max - new_min) / 1e6
                                safe_step_for_div = step_ma2 if (np.isfinite(step_ma2) and step_ma2 > 1e-9) else max(span_ma_new, 1.0)
                                if not np.isfinite(safe_step_for_div) or safe_step_for_div <= 0.0:
                                    safe_step_for_div = max(span_ma_new, 1.0)

                                prev_min = float(st.minimumRimAge)
                                prev_max = float(st.maximumRimAge)
                                prev_n = int(st.rimAgesSampled)
                                prev_runs = list(sample.monteCarloRuns)
                                prev_catalogue = list(getattr(sample, "peak_catalogue", []) or [])

                                # Skip no-op reruns when bounds are unchanged after
                                # clamping (common near the 1 Ma lower bound).
                                if (abs(new_min - prev_min) > 1.0) or (abs(new_max - prev_max) > 1.0):
                                    try:
                                        st.minimumRimAge  = new_min
                                        st.maximumRimAge  = new_max
                                        st.rimAgesSampled = int(max(3, round(span_ma_new / safe_step_for_div) + 1))

                                        # re-run once with widened window
                                        sample.monteCarloRuns = []
                                        sample.peak_catalogue = []
                                        completed, skip_reason = _performRimAgeSampling(signals, sample)
                                        if not completed:
                                            st.minimumRimAge = prev_min
                                            st.maximumRimAge = prev_max
                                            st.rimAgesSampled = prev_n
                                            sample.monteCarloRuns = prev_runs
                                            sample.peak_catalogue = prev_catalogue
                                            return False, skip_reason
                                    except Exception:
                                        st.minimumRimAge = prev_min
                                        st.maximumRimAge = prev_max
                                        st.rimAgesSampled = prev_n
                                        sample.monteCarloRuns = prev_runs
                                        sample.peak_catalogue = prev_catalogue
                                        raise

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
    - For each MC run, constructs a MonteCarloRun over the Pb-loss grid,
      finds the optimal Pb-loss age, and emits ProgressType.SAMPLING.
    """
    sample.monteCarloRuns = []
    sample.peak_catalogue = []
    sampleNameText = f" for '{sample.name}'" if sample.name else ""
    signals.newTask("Sampling Pb-loss age distributions" + sampleNameText + "...")

    settings = sample.calculationSettings
    setattr(settings, "timing_mode", TIMING_MODE)
    setattr(settings, "write_outputs", CDC_WRITE_OUTPUTS)

    eligibleSpots   = [s for s in sample.validSpots if not getattr(s, "reverseDiscordant", False)]
    concordantSpots = [s for s in eligibleSpots if s.concordant]
    discordantSpots = [s for s in eligibleSpots if not s.concordant]

    sample._ks_concordantSpots = list(concordantSpots)
    sample._ks_discordantSpots = list(discordantSpots)

    if not concordantSpots:
        return False, "no concordant spots"
    if not discordantSpots:
        return False, "no discordant spots"
    if len(discordantSpots) <= 2:
        return False, "fewer than 3 discordant spots"

    stabilitySamples = int(settings.monteCarloRuns)
    merge_nearby = bool(getattr(settings, "merge_nearby_peaks", MERGE_NEARBY_PEAKS))
    rng = np.random.default_rng(_seed_from_name(sample.name))

    concordantUPbValues  = np.stack([rng.normal(s.uPbValue,  s.uPbStDev,  stabilitySamples) for s in concordantSpots], axis=1)
    concordantPbPbValues = np.stack([rng.normal(s.pbPbValue, s.pbPbStDev, stabilitySamples) for s in concordantSpots], axis=1)
    discordantUPbValues  = np.stack([rng.normal(s.uPbValue,  s.uPbStDev,  stabilitySamples) for s in discordantSpots], axis=1)
    discordantPbPbValues = np.stack([rng.normal(s.pbPbValue, s.pbPbStDev, stabilitySamples) for s in discordantSpots], axis=1)

    # --------- sampling loop ---------
    per_run_times = []
    t0 = time.perf_counter()
    for j in range(stabilitySamples):
        if signals.halt():
            signals.cancelled()
            return False, "processing halted by user"

        t_run = time.perf_counter()

        run = MonteCarloRun(
            j, sample.name,
            concordantUPbValues[j],  concordantPbPbValues[j],
            discordantUPbValues[j],  discordantPbPbValues[j],
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

def _is_effectively_monotonic(y_curve, delta):
    """
    Return True when the smoothed ensemble curve is effectively monotonic.

    Tiny wiggles are ignored using a derivative epsilon scaled by ensemble
    dynamic range (delta), so "boundary optima" are abstained rather than
    promoted to discrete peaks.
    """
    y = np.asarray(y_curve, float)
    if y.size < 4 or (not np.isfinite(y).any()):
        return True

    dy = np.diff(y)
    eps = max(1e-12, float(MONO_DY_EPS_FRAC) * max(float(delta), 1e-12))
    sgn = np.zeros_like(dy, dtype=int)
    sgn[dy > eps] = 1
    sgn[dy < -eps] = -1
    sgn = sgn[sgn != 0]
    if sgn.size == 0:
        return True

    turns = int(np.sum(sgn[1:] != sgn[:-1]))
    return turns <= int(MONO_MAX_TURNS)

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

def _recompute_winner_support(rows, optima_ma, ages_ma, min_support=None):
    """
    Recompute support as winner-vote fraction from per-run optima.

    Each run contributes to at most one peak:
      1) prefer peaks whose CI contains that run optimum,
      2) tie-break by nearest peak age,
      3) if none contain it, allow nearest peak within a small cap distance.
    """
    if not rows:
        return rows

    rows = sorted([dict(r) for r in rows], key=lambda rr: float(rr["age_ma"]))
    centers = np.array([float(r["age_ma"]) for r in rows], float)
    ci_lo = np.array([float(r["ci_low"]) for r in rows], float)
    ci_hi = np.array([float(r["ci_high"]) for r in rows], float)
    counts = np.zeros(len(rows), float)

    opts = np.asarray(optima_ma, float)
    opts = opts[np.isfinite(opts)]
    if opts.size == 0:
        return rows

    step = float(np.median(np.diff(ages_ma))) if np.asarray(ages_ma).size >= 2 else 5.0
    cap = max(3.0 * step, 30.0)

    for o in opts:
        in_ci = np.where((o >= ci_lo) & (o <= ci_hi))[0]
        if in_ci.size > 0:
            loc = np.argmin(np.abs(centers[in_ci] - o))
            counts[int(in_ci[loc])] += 1.0
            continue

        j = int(np.argmin(np.abs(centers - o)))
        if abs(float(centers[j]) - float(o)) <= cap:
            counts[j] += 1.0

    denom = float(max(opts.size, 1))
    out = []
    for i, r in enumerate(rows):
        winner_sup = float(counts[i] / denom)
        direct_sup = float(r.get("direct_support", r.get("support", float("nan"))))
        if not np.isfinite(direct_sup):
            direct_sup = winner_sup
        rr = dict(
            r,
            direct_support=direct_sup,
            winner_support=winner_sup,
            support=direct_sup,   # keep legacy field as reproducibility support
        )
        if (min_support is not None) and (direct_sup < float(min_support)):
            continue
        out.append(rr)

    out = sorted(out, key=lambda rr: float(rr["age_ma"]))
    for i, rr in enumerate(out, 1):
        rr["peak_no"] = i
    return out


def _support_score(row, mode):
    """Return support score used for inclusion filtering."""
    mode = str(mode).strip().upper()
    winner = float(row.get("winner_support", row.get("support", 0.0)))
    direct = float(row.get("direct_support", row.get("support", 0.0)))
    if mode == "DIRECT":
        return direct
    if mode == "MAX":
        return max(winner, direct)
    return winner  # legacy


def _apply_support_filter(rows, min_support, mode):
    """
    Apply configurable inclusion filtering on support:
      - WINNER: winner_support
      - DIRECT: direct_support
      - MAX: max(direct_support, winner_support)
    """
    if not rows:
        return rows
    out = []
    for r in rows:
        score = float(_support_score(r, mode))
        rr = dict(r, filter_support=score)
        if score >= float(min_support):
            out.append(rr)
    out = sorted(out, key=lambda rr: float(rr["age_ma"]))
    for i, rr in enumerate(out, 1):
        rr["peak_no"] = i
    return out

def _step_ma_from_grid(ages_ma):
    ages_ma = np.asarray(ages_ma, float)
    if ages_ma.size >= 2:
        step = float(np.median(np.diff(ages_ma)))
        if np.isfinite(step) and step > 0.0:
            return step
    return 5.0


def _row_match_index(target_row, candidates, used, tol_ma):
    """Return the unmatched candidate index closest in age to target_row within tol_ma."""
    tgt = float(target_row.get("age_ma", np.nan))
    if not np.isfinite(tgt):
        return None
    best_i = None
    best_d = None
    for i, row in enumerate(candidates):
        if used[i]:
            continue
        age = float(row.get("age_ma", np.nan))
        if not np.isfinite(age):
            continue
        d = abs(age - tgt)
        if d <= tol_ma and (best_d is None or d < best_d):
            best_i = i
            best_d = d
    return best_i


def _append_rejected_peak(rejected_rows, row, reason_code):
    """Append one rejected candidate row unless a near-identical age is already recorded."""
    age = float(row.get("age_ma", np.nan))
    if not np.isfinite(age):
        return
    for rr in rejected_rows:
        if abs(float(rr.get("age_ma", np.nan)) - age) <= 1e-6:
            return
    direct = float(row.get("direct_support", row.get("support", np.nan)))
    winner = float(row.get("winner_support", row.get("support", np.nan)))
    rejected_rows.append(
        dict(
            age_ma=age,
            ci_low=float(row.get("ci_low", np.nan)),
            ci_high=float(row.get("ci_high", np.nan)),
            direct_support=direct,
            winner_support=winner,
            reason=str(reason_code),
        )
    )


def _capture_rejected_step(before_rows, after_rows, rejected_rows, reason_code, ages_ma):
    """Record rows present in before_rows that disappear in after_rows as rejected with reason_code."""
    if not before_rows:
        return
    after_rows = [dict(r) for r in (after_rows or [])]
    used = [False] * len(after_rows)
    tol = max(0.51 * _step_ma_from_grid(ages_ma), 1e-6)
    for row in before_rows:
        j = _row_match_index(row, after_rows, used, tol)
        if j is None:
            _append_rejected_peak(rejected_rows, row, reason_code)
        else:
            used[j] = True

def _apply_boundary_dominance_guard(rows, optima_ma, ages_ma):
    """
    Suppress near-edge peaks when per-run optima are overwhelmingly boundary-dominated.

    This targets failure modes where a broad/monotonic surface produces a spurious
    near-edge local hump with high reproducibility, especially after cluster stacking.
    """
    if (not rows) or (len(rows) == 0):
        return rows, None

    ages_ma = np.asarray(ages_ma, float)
    opts = np.asarray(optima_ma, float)
    opts = opts[np.isfinite(opts)]
    if ages_ma.size < 2 or opts.size == 0:
        return rows, None

    lo = float(ages_ma[0])
    hi = float(ages_ma[-1])
    step = float(np.median(np.diff(ages_ma)))
    if (not np.isfinite(step)) or step <= 0.0:
        step = max((hi - lo) / max(len(ages_ma) - 1, 1), 1.0)

    # Boundary-dominance measured from run-level optima.
    edge_band = max(1.0 * step, 10.0)
    lo_frac = float(np.mean(opts <= (lo + edge_band)))
    hi_frac = float(np.mean(opts >= (hi - edge_band)))

    side = None
    edge_frac = 0.0
    if lo_frac >= hi_frac:
        side, edge_frac = "low", lo_frac
    else:
        side, edge_frac = "high", hi_frac

    # Require substantial boundary pile-up before suppressing.
    if edge_frac < 0.70:
        return rows, None

    opt_med = float(np.median(opts))
    span = max(hi - lo, step)
    # Keep edge zones scale-aware so genuinely young events are not suppressed
    # just because absolute fallback floors dominate small modelling windows.
    near_edge_zone = min(max(8.0 * step, 0.08 * span), 0.25 * span)
    far_from_edge_zone = min(max(5.0 * step, 0.05 * span), 0.20 * span)
    strict_boundary_mode = edge_frac >= 0.85

    out = []
    for r in rows:
        age = float(r.get("age_ma", np.nan))
        if not np.isfinite(age):
            continue

        support = float(r.get("support", 0.0))
        direct = float(r.get("direct_support", support))
        winner = float(r.get("winner_support", support))
        if not np.isfinite(direct):
            direct = support
        if not np.isfinite(winner):
            winner = support

        if side == "low" and age <= (lo + near_edge_zone):
            mismatch = (
                (opt_med <= (lo + edge_band))
                and ((age - lo) >= far_from_edge_zone)
                and (winner < max(0.45, 0.65 * direct))
            )
            if strict_boundary_mode or mismatch:
                continue
        if side == "high" and age >= (hi - near_edge_zone):
            mismatch = (
                (opt_med >= (hi - edge_band))
                and ((hi - age) >= far_from_edge_zone)
                and (winner < max(0.45, 0.65 * direct))
            )
            if strict_boundary_mode or mismatch:
                continue
        out.append(dict(r))

    # If everything was near-edge under boundary dominance, abstain.
    if len(out) == 0:
        return [], "boundary_dominated_surface"
    return out, None


def _plateau_dedupe_rows(rows, ages_ma):
    """
    Collapse near-identical peaks that sit on the same broad/flat crest.
    This is lighter than full peak merging and mainly removes duplicate picks
    emitted from adjacent cluster surfaces.
    """
    if (not rows) or len(rows) <= 1:
        return rows

    rows = sorted([dict(r) for r in rows], key=lambda rr: float(rr["age_ma"]))
    step = float(np.median(np.diff(ages_ma))) if np.asarray(ages_ma).size >= 2 else 5.0
    near_ma = max(float(PLATEAU_DEDUPE_RADIUS_STEPS) * step, 20.0)
    min_ov = float(PLATEAU_DEDUPE_MIN_OVERLAP_FRAC)

    def _width(rr):
        return max(0.0, float(rr["ci_high"]) - float(rr["ci_low"]))

    def _overlap_frac(a, b):
        lo = max(float(a["ci_low"]), float(b["ci_low"]))
        hi = min(float(a["ci_high"]), float(b["ci_high"]))
        ov = max(0.0, hi - lo)
        wa, wb = _width(a), _width(b)
        denom = max(min(wa, wb), 1e-9)
        return ov / denom

    def _score(rr):
        win = float(rr.get("winner_support", rr.get("support", 0.0)))
        direct = float(rr.get("direct_support", rr.get("support", 0.0)))
        width = _width(rr)
        return (win, direct, -width)

    deduped = [dict(rows[0])]
    for rr in rows[1:]:
        prev = deduped[-1]
        sep = abs(float(rr["age_ma"]) - float(prev["age_ma"]))
        same_crest = (sep <= near_ma) and (_overlap_frac(prev, rr) >= min_ov)
        if same_crest:
            if _score(rr) > _score(prev):
                deduped[-1] = dict(rr)
        else:
            deduped.append(dict(rr))

    for i, rr in enumerate(deduped, 1):
        rr["peak_no"] = i
    return deduped


def _single_crest_fallback_row(ages_ma, S_curve, optima_ma, min_support):
    """
    Conservative fallback when strict peak gating abstains:
      - require one strong interior crest on the displayed ensemble curve,
      - require non-trivial run-optima concentration in that crest window.
    Returns a single peak row dict or None.
    """
    x = np.asarray(ages_ma, float)
    y = np.asarray(S_curve, float)
    if x.size < 7 or y.size != x.size:
        return None

    finite = np.isfinite(y)
    if not np.any(finite):
        return None
    y = np.where(finite, y, np.nan)

    q5, q95 = np.nanpercentile(y, [5, 95])
    delta = float(max(q95 - q5, 0.0))
    if delta < max(float(ENS_DELTA_MIN), 1e-6):
        return None

    step = float(np.median(np.diff(x))) if x.size >= 2 else 1.0
    if (not np.isfinite(step)) or step <= 0.0:
        step = max((float(x[-1]) - float(x[0])) / max(int(x.size) - 1, 1), 1.0)
    lo_age, hi_age = float(x[0]), float(x[-1])
    span = max(hi_age - lo_age, step)
    edge_margin = min(max(6.0 * step, 0.07 * span), 0.30 * span)

    # Simple local maxima detector on the smoothed ensemble curve.
    loc = np.where((y[1:-1] >= y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
    if loc.size == 0:
        try:
            loc = np.array([int(np.nanargmax(y))], dtype=int)
        except Exception:
            return None

    interior = [int(j) for j in loc if (x[j] > (lo_age + edge_margin)) and (x[j] < (hi_age - edge_margin))]
    if not interior:
        return None

    j = max(interior, key=lambda idx: float(y[idx]))
    left_min = float(np.nanmin(y[: j + 1]))
    right_min = float(np.nanmin(y[j:]))
    left_lift = float(y[j] - left_min)
    right_lift = float(y[j] - right_min)
    prom_balanced = float(y[j] - max(left_min, right_min))
    prom_one_sided = float(max(left_lift, right_lift))
    # Accept broad single-crest surfaces where absolute prominence is modest
    # compared to full-window delta, but still clearly above local roughness.
    rough = float(np.nanmedian(np.abs(np.diff(y)))) if y.size >= 3 else 0.0
    prom_min = max(0.03 * delta, 3.0 * rough, 0.008)
    weak_side_min = max(rough, 0.002)
    if prom_one_sided < prom_min:
        return None
    if min(left_lift, right_lift) < weak_side_min:
        return None

    # Half-prominence window around the crest.
    prom_for_window = max(prom_balanced, prom_min)
    half_level = float(y[j] - 0.5 * prom_for_window)
    jl = int(j)
    while jl > 0 and float(y[jl]) >= half_level:
        jl -= 1
    jr = int(j)
    n = int(y.size)
    while jr < (n - 1) and float(y[jr]) >= half_level:
        jr += 1

    lo_win = float(x[max(jl, 0)])
    hi_win = float(x[min(jr, n - 1)])
    if hi_win <= lo_win:
        pad = max(2.0 * step, 0.04 * span)
        lo_win = max(lo_age, float(x[j]) - pad)
        hi_win = min(hi_age, float(x[j]) + pad)

    opts = np.asarray(optima_ma, float)
    opts = opts[np.isfinite(opts)]
    if opts.size == 0:
        return None

    # Do not synthesize a fallback peak when run-level optima are dominated by
    # a modelling-window boundary (classic fan-to-zero / floor artefact).
    edge_band = max(5.0 * step, 40.0)
    lo_frac = float(np.mean(opts <= (lo_age + edge_band)))
    hi_frac = float(np.mean(opts >= (hi_age - edge_band)))
    if max(lo_frac, hi_frac) >= 0.70:
        return None

    in_win = opts[(opts >= lo_win) & (opts <= hi_win)]
    winner_support = float(in_win.size / float(max(opts.size, 1)))
    if winner_support < float(min_support):
        return None

    if in_win.size >= 3:
        ci_low, ci_high = np.nanpercentile(in_win, [2.5, 97.5])
        ci_low = float(max(ci_low, lo_age))
        ci_high = float(min(ci_high, hi_age))
    else:
        ci_low, ci_high = lo_win, hi_win

    if (ci_high - ci_low) < step:
        ci_low = max(lo_age, float(x[j]) - step)
        ci_high = min(hi_age, float(x[j]) + step)
    age = float(x[j])
    if not (ci_low <= age <= ci_high):
        ci_low = min(ci_low, age)
        ci_high = max(ci_high, age)

    return dict(
        age_ma=age,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        support=winner_support,
        direct_support=winner_support,
        winner_support=winner_support,
        selection="fallback",
        peak_no=1,
    )


def _snap_rows_to_curve(rows, ages_ma, S_view):
    """
    Keep accepted catalogue rows unchanged for reporting, but snap plotted marker
    ages to local crests of the final displayed curve.
    """
    if not rows:
        return []

    x = np.asarray(ages_ma, float)
    y = np.asarray(S_view, float)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return [dict(r) for r in rows]

    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return [dict(r) for r in rows]

    # Candidate local maxima on the displayed curve.
    core = np.where(
        finite[1:-1] &
        (y[1:-1] >= y[:-2]) &
        (y[1:-1] >= y[2:])
    )[0] + 1
    if core.size == 0:
        idx_all = np.where(finite)[0]
        if idx_all.size:
            core = np.array([int(idx_all[np.nanargmax(y[idx_all])])], dtype=int)
        else:
            return [dict(r) for r in rows]

    # Deduplicate adjacent maxima (plateaus) by keeping the midpoint of the
    # flat crest when multiple adjacent nodes share the same local maximum.
    def _representative_max(run_idx):
        run_idx = np.asarray(run_idx, dtype=int)
        if run_idx.size == 0:
            raise ValueError("run_idx must not be empty")
        y_run = y[run_idx]
        y_max = float(np.nanmax(y_run))
        tied = run_idx[np.isclose(y_run, y_max, rtol=1e-12, atol=1e-15)]
        if tied.size:
            return int(tied[tied.size // 2])
        return int(run_idx[int(np.nanargmax(y_run))])

    maxima = []
    run = [int(core[0])]
    for idx in core[1:]:
        idx = int(idx)
        if idx == run[-1] + 1:
            run.append(idx)
        else:
            maxima.append(_representative_max(run))
            run = [idx]
    if run:
        maxima.append(_representative_max(run))
    maxima = np.asarray(maxima, dtype=int)

    out = []
    used = np.zeros(maxima.size, dtype=bool)
    step = float(np.median(np.diff(x))) if x.size >= 2 else 1.0
    lo_grid = float(np.nanmin(x[finite]))
    hi_grid = float(np.nanmax(x[finite]))

    # Assign each displayed row to the nearest available displayed-curve crest.
    order = np.argsort([float(dict(r).get("age_ma", np.nan)) for r in rows])
    snapped = [None] * len(rows)
    for ii in order:
        rr = dict(rows[ii])
        a0 = float(rr.get("age_ma", np.nan))
        try:
            lo_old = float(rr.get("ci_low", np.nan))
            hi_old = float(rr.get("ci_high", np.nan))
        except Exception:
            lo_old, hi_old = np.nan, np.nan

        if maxima.size > 0 and np.any(~used):
            avail = np.where(~used)[0]
            j_idx = int(avail[np.argmin(np.abs(x[maxima[avail]] - a0))])
            j = int(maxima[j_idx])
            used[j_idx] = True
        elif maxima.size > 0:
            j = int(maxima[np.argmin(np.abs(x[maxima] - a0))])
        else:
            idx_all = np.where(finite)[0]
            j = int(idx_all[np.argmin(np.abs(x[idx_all] - a0))])

        a_new = float(x[j])
        rr["age_ma"] = a_new

        width = hi_old - lo_old if (np.isfinite(lo_old) and np.isfinite(hi_old)) else np.nan
        if (not np.isfinite(width)) or (width <= 0.0):
            width = 2.0 * step
        lo_new = max(lo_grid, a_new - 0.5 * width)
        hi_new = min(hi_grid, a_new + 0.5 * width)
        if (hi_new - lo_new) < step:
            lo_new = max(lo_grid, a_new - step)
            hi_new = min(hi_grid, a_new + step)
        rr["ci_low"] = float(min(lo_new, a_new))
        rr["ci_high"] = float(max(hi_new, a_new))
        snapped[ii] = rr

    out = [r for r in snapped if isinstance(r, dict)]
    return out


def _raw_optimum_age_ma(run) -> float:
    """
    Return the per-run RAW optimum age (Ma) from `_raw_statistics_by_pb_loss_age`.

    Falls back to the penalised optimum if RAW data are unavailable.
    """
    raw_map = getattr(run, "_raw_statistics_by_pb_loss_age", None)
    if isinstance(raw_map, dict) and raw_map:
        ages = np.array(sorted(raw_map.keys()), float)
        dvals = np.array([raw_map[a].test_statistics[0] for a in ages], float)
        dvals = np.where(np.isfinite(dvals), dvals, np.inf)
        if ages.size and np.isfinite(dvals).any():
            idx = _findOptimalIndex(dvals.tolist())
            return float(ages[idx] / 1e6)

    pen = float(getattr(run, "optimal_pb_loss_age", np.nan))
    return float(pen / 1e6) if np.isfinite(pen) else float("nan")


def _stack_goodness_from_stats_attr(runs, ages_y, stats_attr: str, which: str = "pen") -> np.ndarray:
    """
    Build an R×G goodness matrix from a run-level stats map attribute.

    Parameters
    ----------
    runs : list[MonteCarloRun]
        Monte Carlo runs.
    ages_y : array_like
        Age grid in YEARS.
    stats_attr : str
        Attribute name on each run that maps age_years -> statistics object.
    which : {'raw', 'pen'}
        'raw' -> 1 - KS D ; 'pen' -> 1 - penalized score.
    """
    ages_y = np.asarray(ages_y, float)
    R = len(runs)
    G = len(ages_y)
    S = np.full((R, G), np.nan, float)

    for r_i, run in enumerate(runs):
        stats_map = getattr(run, stats_attr, None)
        if not isinstance(stats_map, dict) or not stats_map:
            stats_map = getattr(run, "statistics_by_pb_loss_age", None)
        if not isinstance(stats_map, dict) or not stats_map:
            continue

        arr = np.full(G, np.nan, float)
        for g_i, age in enumerate(ages_y):
            st = stats_map.get(float(age))
            if st is None:
                continue
            if which == "raw":
                arr[g_i] = 1.0 - float(st.test_statistics[0])
            else:
                ds = float(st.score)
                ds = min(1.0, max(0.0, ds))
                arr[g_i] = 1.0 - ds
        S[r_i] = arr

    return S


def _optimum_age_ma_from_stats_attr(run, stats_attr: str, which: str = "pen") -> float:
    """
    Return per-run optimum age (Ma) from a run-level stats map attribute.
    """
    stats_map = getattr(run, stats_attr, None)
    if not isinstance(stats_map, dict) or not stats_map:
        stats_map = getattr(run, "statistics_by_pb_loss_age", None)
    if not isinstance(stats_map, dict) or not stats_map:
        return float("nan")

    ages = np.array(sorted(stats_map.keys()), float)
    if ages.size == 0:
        return float("nan")

    if which == "raw":
        vals = np.array([stats_map[a].test_statistics[0] for a in ages], float)
    else:
        vals = np.array([stats_map[a].score for a in ages], float)
    vals = np.where(np.isfinite(vals), vals, np.inf)
    if not np.isfinite(vals).any():
        return float("nan")

    idx = _findOptimalIndex(vals.tolist())
    return float(ages[idx] / 1e6)


def _optimum_stat_from_stats_attr(run, stats_attr: str, which: str = "pen"):
    """
    Return the run-level optimum statistics object from a chosen stats map.
    """
    stats_map = getattr(run, stats_attr, None)
    if not isinstance(stats_map, dict) or not stats_map:
        stats_map = getattr(run, "statistics_by_pb_loss_age", None)
    if not isinstance(stats_map, dict) or not stats_map:
        return None

    ages = np.array(sorted(stats_map.keys()), float)
    if ages.size == 0:
        return None

    if which == "raw":
        vals = np.array([stats_map[a].test_statistics[0] for a in ages], float)
    else:
        vals = np.array([stats_map[a].score for a in ages], float)
    vals = np.where(np.isfinite(vals), vals, np.inf)
    if not np.isfinite(vals).any():
        return None

    idx = _findOptimalIndex(vals.tolist())
    return stats_map[float(ages[idx])]


def _median_best_from_runs(runs, stats_attr: str, which: str = "pen") -> float:
    """
    Median run-level best dissimilarity from a stats map attribute.
    Lower is better.
    """
    vals = []
    for run in runs:
        stats_map = getattr(run, stats_attr, None)
        if not isinstance(stats_map, dict) or not stats_map:
            continue
        ages = sorted(stats_map.keys())
        if not ages:
            continue
        if which == "raw":
            arr = np.array([stats_map[a].test_statistics[0] for a in ages], float)
        else:
            arr = np.array([stats_map[a].score for a in ages], float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            vals.append(float(np.min(arr)))
    if not vals:
        return float("nan")
    return float(np.median(np.asarray(vals, float)))


def _build_global_catalogue_rows(sample_name: str,
                                 tier: str,
                                 ages_ma: np.ndarray,
                                 S_runs: np.ndarray,
                                 Smed: np.ndarray,
                                 *,
                                 smf: float,
                                 merge_nearby: bool,
                                 pickable: bool,
                                 optima_ma: np.ndarray,
                                 diagnostic_rows: Optional[List[Dict]] = None):
    """
    Build strict global ensemble rows for one surface, with one relaxed-prom fallback.
    """
    if (not pickable) or (Smed.size == 0):
        return []

    diag_rows: List[Dict] = []

    rows = build_ensemble_catalogue(
        sample_name, tier, ages_ma, S_runs,
        orientation="max", smooth_frac=smf,
        f_d=FD_DIST_FRAC, f_p=FP_PROM_FRAC, f_v=FV_VALLEY_FRAC, f_w=FW_WIN_FRAC,
        w_min_nodes=3, support_min=FS_SUPPORT, r_min=RMIN_RUNS, f_r=FR_RUN_REL,
        per_run_prom_frac=PER_RUN_PROM_FRAC, per_run_min_dist=PER_RUN_MIN_DIST,
        per_run_min_width=PER_RUN_MIN_WIDTH, per_run_require_full_prom=False,
        pen_ok_mask=None, cand_curve=Smed, height_frac=FH_HEIGHT_FRAC, optima_ma=optima_ma,
        merge_per_hump=merge_nearby, merge_shoulders=merge_nearby,
        diagnostic_rows=diag_rows,
    ) or []

    if not rows:
        diag_rows = []
        rows = build_ensemble_catalogue(
            sample_name, tier, ages_ma, S_runs,
            orientation="max", smooth_frac=smf,
            f_d=FD_DIST_FRAC, f_p=max(0.5 * FP_PROM_FRAC, 0.01),
            f_v=FV_VALLEY_FRAC, f_w=FW_WIN_FRAC,
            w_min_nodes=3, support_min=FS_SUPPORT, r_min=RMIN_RUNS, f_r=FR_RUN_REL,
            per_run_prom_frac=PER_RUN_PROM_FRAC, per_run_min_dist=PER_RUN_MIN_DIST,
            per_run_min_width=PER_RUN_MIN_WIDTH, per_run_require_full_prom=False,
            pen_ok_mask=None, cand_curve=Smed, height_frac=FH_HEIGHT_FRAC, optima_ma=optima_ma,
            merge_per_hump=merge_nearby, merge_shoulders=merge_nearby,
            diagnostic_rows=diag_rows,
        ) or []

    if diagnostic_rows is not None:
        diagnostic_rows.extend(dict(r) for r in diag_rows)

    return rows

def _calculateOptimalAge(signals, sample, progress):
    """
      • UI shows median-of-run-optima with consistent CI.
      • Exports and figures use legacy surface optimum and curve.
    """
    settings, runs = sample.calculationSettings, sample.monteCarloRuns
    if not runs:
        return
    sample.rejected_peak_candidates = []

    merge_nearby = bool(getattr(settings, "merge_nearby_peaks", MERGE_NEARBY_PEAKS))
    abstain_on_monotonic = bool(getattr(settings, "conservative_abstain_on_monotonic", True))
    support_filter_mode = "DIRECT"
    prefer_pen = bool(getattr(settings, "penaliseInvalidAges", False))
    primary_which = "pen" if prefer_pen else "raw"

    # Grid
    ages_y = np.asarray(settings.rimAges(), float)   # years
    ages_ma = ages_y / 1e6
    optima_ma_pen = np.array(
        [_optimum_age_ma_from_stats_attr(r, "_all_statistics_by_pb_loss_age", which="pen") for r in runs],
        float,
    )
    optima_ma_raw = np.array(
        [_optimum_age_ma_from_stats_attr(r, "_all_statistics_by_pb_loss_age", which="raw") for r in runs],
        float,
    )

    # Per-run goodness matrices from the global all-discordants surface.
    S_runs_raw = _stack_goodness_from_stats_attr(
        runs, ages_y, "_all_statistics_by_pb_loss_age", which="raw"
    )
    S_runs_pen = _stack_goodness_from_stats_attr(
        runs, ages_y, "_all_statistics_by_pb_loss_age", which="pen"
    )

    # Smoothing + ensemble curves (only relevant if ensemble mode ON)
    smf = _smooth_frac_for_grid(ages_ma)
    Smed_raw, Delta_raw, _ = robust_ensemble_curve(S_runs_raw, smooth_frac=smf)
    Smed_pen, Delta_pen, _ = robust_ensemble_curve(S_runs_pen, smooth_frac=smf)
    mono_raw = _is_effectively_monotonic(Smed_raw, Delta_raw)
    mono_pen = _is_effectively_monotonic(Smed_pen, Delta_pen)
    ui_surface = str(getattr(settings, "catalogue_surface", CATALOGUE_SURFACE)).strip().upper()
    if ui_surface not in {"RAW", "PEN"}:
        ui_surface = "PEN"
    # Initial display surface; final selection follows the same global surface
    # that produces the accepted rows.
    S_view = Smed_raw if (ui_surface == "RAW") else Smed_pen
    sample.ensemble_surface_flags = dict(
        raw_delta=float(Delta_raw),
        pen_delta=float(Delta_pen),
        raw_monotonic=bool(mono_raw),
        pen_monotonic=bool(mono_pen),
        primary_channel=str(primary_which),
        view_surface_source="global_all",
    )
    sample.ensemble_abstain_reason = None
    raw_pickable = (Delta_raw >= ENS_DELTA_MIN) and ((not abstain_on_monotonic) or (not mono_raw))
    pen_pickable = (Delta_pen >= ENS_DELTA_MIN) and ((not abstain_on_monotonic) or (not mono_pen))

    # ---------- (A) Run-optima median & CI (for UI) ----------
    optima_ma_primary = optima_ma_pen if prefer_pen else optima_ma_raw
    opt_all = np.sort(np.asarray(optima_ma_primary[np.isfinite(optima_ma_primary)] * 1e6, float))
    if opt_all.size == 0:
        if prefer_pen:
            opt_all = np.sort(np.asarray([r.optimal_pb_loss_age for r in runs], float))
        else:
            opt_all = np.sort(np.asarray([_raw_optimum_age_ma(r) * 1e6 for r in runs], float))
    n = opt_all.size
    if n:
        optimalAge_ui = float(np.median(opt_all))  # years
        lower95 = float(opt_all[int(np.floor(0.025 * n))])
        upper95 = float(opt_all[int(np.ceil(0.975 * n)) - 1])
    else:
        optimalAge_ui = lower95 = upper95 = float("nan")
    optimalAge = optimalAge_ui 
    # ---------- (B) Legacy surface optimum (for export/figure) ----------
    # Legacy export curve remains a simple mean dissimilarity-by-age, tied to
    # the active primary channel on the global all-discordants surface.
    S_runs_primary = S_runs_pen if prefer_pen else S_runs_raw
    sum_good = np.nansum(S_runs_primary, axis=0)
    cnt_good = np.sum(np.isfinite(S_runs_primary), axis=0)
    mean_good = np.divide(
        sum_good,
        cnt_good,
        out=np.full_like(sum_good, np.nan, dtype=float),
        where=cnt_good > 0,
    )
    mean_primary = 1.0 - mean_good
    mean_primary = np.where(np.isfinite(mean_primary), mean_primary, np.inf)
    legacy_idx = _findOptimalIndex(mean_primary.tolist())
    optimalAge_legacy = float(ages_y[legacy_idx])  # years (grid node)
    S_legacy_curve = 1.0 - mean_primary  # legacy S(t)
    sample.legacy_surface_optimal_age = optimalAge_legacy

    # ---------- (C) Mean stats at each run’s own optimum ----------
    stats = [_optimum_stat_from_stats_attr(r, "_all_statistics_by_pb_loss_age", which=primary_which) for r in runs]
    stats = [s for s in stats if s is not None]
    if stats:
        meanD = float(np.mean([s.test_statistics[0] for s in stats]))
        pvals = np.asarray([s.test_statistics[1] for s in stats], float)
        p_ok = np.isfinite(pvals)
        meanP = float(np.mean(pvals[p_ok])) if np.any(p_ok) else float("nan")
        meanInv = float(np.mean([s.number_of_invalid_ages for s in stats]))
        if prefer_pen:
            meanSc = float(np.mean([s.score for s in stats]))
        else:
            meanSc = float(np.mean([s.test_statistics[0] for s in stats]))
    else:
        meanD = meanP = meanInv = meanSc = float("nan")

    # ---------- (D) Ensemble OFF branch ----------
    enabled = bool(getattr(settings, "enable_ensemble_peak_picking", False))
    if not enabled:
        # UI point estimate = median of run optima
        optimalAge = optimalAge_ui
        sample.legacy_surface_optimal_age = optimalAge_legacy

        sample.peak_catalogue = []
        try:
            sample.signals.optimalAgeCalculated.emit()
        except Exception:
            pass

        # Plot legacy curve for diagnostics (optional)
        _emit_summedKS(signals, sample, progress, ages_ma, S_legacy_curve, rows_for_ui=[])

        # Export legacy surface optimum + curve (for paper figure)
        if KS_EXPORT_ROOT is not None:
            _export_legacy_ks(
                sample, settings, runs, ages_y,
                D_pen=mean_primary,
                ui_opt_years=optimalAge_ui,
                ui_low95_years=lower95,
                ui_high95_years=upper95,
                run_optima_years=opt_all,
                legacy_opt_years=optimalAge_legacy,
            )


        payload = (optimalAge, lower95, upper95, meanD, meanP, meanInv, meanSc, "—", [])
        try:
            signals.progress(ProgressType.OPTIMAL, 1.0, sample.name, payload)
        except TypeError:
            signals.progress(ProgressType.OPTIMAL, 1.0, sample.name, payload[:7])
        return
    # ------------------------------------------------------------------
    # Build catalogues from one path only: the global all-discordants surface.
    # ------------------------------------------------------------------
    rows_raw: List[Dict] = []
    rows_pen: List[Dict] = []
    rejected_raw_stage: List[Dict] = []
    rejected_pen_stage: List[Dict] = []
    rows_raw = _build_global_catalogue_rows(
        sample.name,
        _infer_tier(sample.name),
        ages_ma,
        S_runs_raw,
        Smed_raw,
        smf=smf,
        merge_nearby=merge_nearby,
        pickable=raw_pickable,
        optima_ma=optima_ma_raw,
        diagnostic_rows=rejected_raw_stage,
    )
    rows_pen = _build_global_catalogue_rows(
        sample.name,
        _infer_tier(sample.name),
        ages_ma,
        S_runs_pen,
        Smed_pen,
        smf=smf,
        merge_nearby=merge_nearby,
        pickable=pen_pickable,
        optima_ma=optima_ma_pen,
        diagnostic_rows=rejected_pen_stage,
    )

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
        if (not raw_pickable) and (not pen_pickable):
            sample.ensemble_abstain_reason = "flat_or_monotonic_surface"

    # Choose which to DISPLAY in the UI strictly from the same ensemble surface
    # that produced the accepted rows.
    if (ui_surface == "RAW") and rows_raw:
        view_which = "raw"
        rows_for_ui = rows_raw
        S_view = Smed_raw
        rejected_stage_ui = rejected_raw_stage
    elif rows_pen:
        view_which = "pen"
        rows_for_ui = rows_pen
        S_view = Smed_pen
        rejected_stage_ui = rejected_pen_stage
    else:
        rows_for_ui = []
        if ui_surface == "RAW":
            view_which = "raw"
            S_view = Smed_raw
            rejected_stage_ui = rejected_raw_stage
        else:
            view_which = "pen"
            S_view = Smed_pen
            rejected_stage_ui = rejected_pen_stage

    for _rows in (rows_raw, rows_pen, rows_for_ui):
        for _r in _rows:
            _r.setdefault("selection", "strict")

    _ensure_output_dirs()

    rejected_rows: List[Dict] = [dict(r) for r in (rejected_stage_ui or [])]

    if merge_nearby:
        pre_merge_ui = [dict(r) for r in rows_for_ui]
        rows_for_ui = _collapse_ci_clusters(rows_for_ui)
        rows_raw    = _collapse_ci_clusters(rows_raw)
        rows_pen    = _collapse_ci_clusters(rows_pen)
        _capture_rejected_step(pre_merge_ui, rows_for_ui, rejected_rows, "merged_overlapping_candidates", ages_ma)

    # Recompute winner-vote support from per-run optima while preserving
    # direct per-run peak support as the primary "support" metric.
    support_floor = max(float(FS_SUPPORT), 0.03)
    optima_ma_raw_rows = optima_ma_raw
    optima_ma_pen_rows = optima_ma_pen
    rows_raw = _recompute_winner_support(rows_raw, optima_ma_raw_rows, ages_ma, min_support=None)
    rows_pen = _recompute_winner_support(rows_pen, optima_ma_pen_rows, ages_ma, min_support=None)
    if ui_surface == "RAW":
        optima_ma_ui_vote = optima_ma_raw_rows
    else:
        optima_ma_ui_vote = optima_ma_pen_rows
    rows_for_ui = _recompute_winner_support(rows_for_ui, optima_ma_ui_vote, ages_ma, min_support=None)
    rows_raw = _apply_support_filter(rows_raw, support_floor, support_filter_mode)
    rows_pen = _apply_support_filter(rows_pen, support_floor, support_filter_mode)
    pre_support_ui = [dict(r) for r in rows_for_ui]
    rows_for_ui = _apply_support_filter(rows_for_ui, support_floor, support_filter_mode)
    _capture_rejected_step(pre_support_ui, rows_for_ui, rejected_rows, "low_support", ages_ma)

    # In no-merge mode, remove near-identical plateau duplicates and re-vote so
    # winner support reflects the deduped set.
    if (not merge_nearby) and PLATEAU_DEDUPE:
        rows_raw = _plateau_dedupe_rows(rows_raw, ages_ma)
        rows_pen = _plateau_dedupe_rows(rows_pen, ages_ma)
        pre_dedupe_ui = [dict(r) for r in rows_for_ui]
        rows_for_ui = _plateau_dedupe_rows(rows_for_ui, ages_ma)
        _capture_rejected_step(pre_dedupe_ui, rows_for_ui, rejected_rows, "plateau_duplicate", ages_ma)
        rows_raw = _recompute_winner_support(rows_raw, optima_ma_raw_rows, ages_ma, min_support=None)
        rows_pen = _recompute_winner_support(rows_pen, optima_ma_pen_rows, ages_ma, min_support=None)
        rows_for_ui = _recompute_winner_support(rows_for_ui, optima_ma_ui_vote, ages_ma, min_support=None)
        rows_raw = _apply_support_filter(rows_raw, support_floor, support_filter_mode)
        rows_pen = _apply_support_filter(rows_pen, support_floor, support_filter_mode)
        pre_support_ui = [dict(r) for r in rows_for_ui]
        rows_for_ui = _apply_support_filter(rows_for_ui, support_floor, support_filter_mode)
        _capture_rejected_step(pre_support_ui, rows_for_ui, rejected_rows, "low_support", ages_ma)

    # Boundary/fallback guards follow the same surface used for reporting.
    optima_ma_display = optima_ma_raw_rows if view_which == "raw" else optima_ma_pen_rows

    # Boundary-dominance guard: avoid reporting near-edge strict peaks when
    # run-level optima overwhelmingly collapse to a boundary.
    pre_boundary_ui = [dict(r) for r in rows_for_ui]
    rows_for_ui, boundary_reason = _apply_boundary_dominance_guard(rows_for_ui, optima_ma_display, ages_ma)
    if boundary_reason is not None:
        _capture_rejected_step(pre_boundary_ui, rows_for_ui, rejected_rows, boundary_reason, ages_ma)
    if boundary_reason is not None:
        rows_raw = []
        rows_pen = []
        sample.ensemble_abstain_reason = boundary_reason

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
        pre_clean_ui = [dict(r) for r in rows_for_ui]
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
                if float(r.get("filter_support", r.get("support", 0.0))) >= max(support_floor, 0.12):
                    lo, hi = a - step, a + step
                else:
                    continue
            cleaned.append(dict(r, ci_low=lo, ci_high=hi))
        rows_for_ui = cleaned
        _capture_rejected_step(pre_clean_ui, rows_for_ui, rejected_rows, "edge_degenerate_ci", ages_ma)

    for r in rows_for_ui:
        a = float(r["age_ma"])
        if not (float(r["ci_low"]) <= a <= float(r["ci_high"])):
            r["ci_low"]  = min(float(r["ci_low"]),  a)
            r["ci_high"] = max(float(r["ci_high"]), a)

    # ---- Drop peaks with absurdly wide CIs (essentially the entire grid) ----
    if rows_for_ui:
        total_span = float(ages_ma[-1] - ages_ma[0])
        MAX_CI_FRAC = 0.5  # drop peaks whose CI spans >50% of the modelling window

        pre_width_ui = [dict(r) for r in rows_for_ui]
        filtered = []
        for r in rows_for_ui:
            width = float(r["ci_high"] - r["ci_low"])
            if width > MAX_CI_FRAC * total_span:
                continue
            filtered.append(r)
        rows_for_ui = filtered
        _capture_rejected_step(pre_width_ui, rows_for_ui, rejected_rows, "wide_ci", ages_ma)

        # Keep rows_raw/rows_pen in sync with what we actually show
        def _keep_same(rows, keep):
            if not rows or not keep:
                return []
            keep_ages = {float(r["age_ma"]) for r in keep}
            return [r for r in rows if float(r.get("age_ma", float("nan"))) in keep_ages]

        rows_raw = _keep_same(rows_raw, rows_for_ui)
        rows_pen = _keep_same(rows_pen, rows_for_ui)

    # Conservative fallback for clear interior single-crest surfaces.
    if not rows_for_ui:
        fb = _single_crest_fallback_row(
            ages_ma,
            S_view,
            optima_ma_display,
            min_support=max(float(support_floor), 0.10),
        )
        if fb is not None:
            rows_for_ui = [dict(fb)]
            if ui_surface == "RAW":
                rows_raw = [dict(fb)]
                rows_pen = []
            else:
                rows_pen = [dict(fb)]
                rows_raw = []
            sample.ensemble_abstain_reason = None

    # Conservative display/reporting: if no peaks survive, annotate as unresolved.
    if not rows_for_ui:
        if sample.ensemble_abstain_reason is None:
            sample.ensemble_abstain_reason = "no_supported_peaks"

    if isinstance(getattr(sample, "ensemble_surface_flags", None), dict):
        sample.ensemble_surface_flags["view_surface_source"] = "global_all"

    # Keep the top-panel heatmap on the same source as the final reported curve.
    for run in runs:
        run.createHeatmapData(
            settings.minimumRimAge,
            settings.maximumRimAge,
            config.HEATMAP_RESOLUTION,
        )

    # Keep only candidates that were not accepted in final rows_for_ui.
    if rejected_rows:
        used = [False] * len(rows_for_ui)
        tol = max(0.51 * _step_ma_from_grid(ages_ma), 1e-6)
        kept_rejected: List[Dict] = []
        for rr in rejected_rows:
            j = _row_match_index(rr, rows_for_ui, used, tol)
            if j is None:
                kept_rejected.append(rr)
            else:
                used[j] = True
        rejected_rows = sorted(kept_rejected, key=lambda r: float(r.get("age_ma", np.nan)))
    sample.rejected_peak_candidates = rejected_rows

    # Renumber peak_no after filtering so CSV doesn't have gaps
    for i, r in enumerate(rows_for_ui, 1):
        r["peak_no"] = i
    for i, r in enumerate(rows_raw, 1):
        r["peak_no"] = i
    for i, r in enumerate(rows_pen, 1):
        r["peak_no"] = i

    # Export rows_for_ui/rows_raw/rows_pen are final
    if CDC_WRITE_OUTPUTS:
        _append_catalogue_rows(sample.name, rows_pen, dest_path=CATALOGUE_CSV_PEN)
        _append_catalogue_rows(sample.name, rows_raw, dest_path=CATALOGUE_CSV_RAW)
        _write_npz_diagnostics(
            sample_name=sample.name,
            ages_ma=ages_ma,
            ages_y=ages_y,
            runs=runs,
            S_runs_raw=S_runs_raw,
            S_runs_pen=S_runs_pen,
            Smed_raw=Smed_raw,
            Smed_pen=Smed_pen,
            S_view=S_view,
            rows_for_ui=rows_for_ui,
        )

    # Plot rows come from final accepted rows, with marker ages snapped to the
    # final displayed curve for visual consistency.
    rows_for_plot = _snap_rows_to_curve(rows_for_ui, ages_ma, S_view)

    # Publish to UI
    catalogue_legacy = [(r["age_ma"], r["ci_low"], r["ci_high"], r["support"]) for r in rows_for_ui]

    red_peaks = np.asarray([m for m, *_ in catalogue_legacy], float)
    peak_str  = fmt_peak_stats(catalogue_legacy) if catalogue_legacy else "—"

    plot_peaks = np.asarray([float(r.get("age_ma", np.nan)) for r in rows_for_plot], float)
    plot_ci_low = np.asarray([float(r.get("ci_low", np.nan)) for r in rows_for_plot], float)
    plot_ci_high = np.asarray([float(r.get("ci_high", np.nan)) for r in rows_for_plot], float)
    sample.summedKS_peaks_Ma = plot_peaks
    sample.summedKS_ci_low_Ma = plot_ci_low
    sample.summedKS_ci_high_Ma = plot_ci_high
    sample.peak_uncertainty_str = peak_str
    detailed_catalogue = [
        dict(
            sample=sample.name,
            peak_no=i + 1,
            ci_low=lo,
            age_ma=med,
            ci_high=hi,
            support=sup,
            direct_support=float(r.get("direct_support", sup)),
            winner_support=float(r.get("winner_support", sup)),
            selection=r.get("selection", "strict"),
        )
        for i, (r, (med, lo, hi, sup)) in enumerate(zip(rows_for_ui, catalogue_legacy))
    ]
    sample.peak_catalogue = detailed_catalogue

    _emit_summedKS(signals, sample, progress, ages_ma, S_view, rows_for_plot)

    # Preserve the legacy KS exports used by Fig. 07 even when the ensemble
    # catalogue is enabled for reporting.
    if KS_EXPORT_ROOT is not None:
        _export_legacy_ks(
            sample,
            settings,
            runs,
            ages_y,
            ui_opt_years=optimalAge_ui,
            ui_low95_years=lower95,
            ui_high95_years=upper95,
            run_optima_years=opt_all,
            legacy_opt_years=optimalAge_legacy,
        )

    try:
        sample.signals.optimalAgeCalculated.emit()
    except Exception:
        pass

    payload = (
        optimalAge,
        lower95,
        upper95,
        meanD,
        meanP,
        meanInv,
        meanSc,
        peak_str,
        detailed_catalogue,
        {"rejected_peak_candidates": list(rejected_rows or [])},
    )
    try:
        signals.progress(ProgressType.OPTIMAL, 1.0, sample.name, payload)
    except TypeError:
        signals.progress(ProgressType.OPTIMAL, 1.0, sample.name, payload[:7])
